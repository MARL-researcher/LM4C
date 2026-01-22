import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualWorldModel(nn.Module):
    def __init__(self, args):
        super(ResidualWorldModel, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        # Calculate total state dimension
        self.state_dim = int(np.prod(args.state_shape))
        
        # Hyperparameters
        self.hidden_dim = args.rnn_hidden_dim
        self.action_embed_dim = args.action_embed_dim          # Compress one-hot actions to dense vectors
        
        # 1. Action Encoder: Maps discrete/one-hot actions to dense embeddings
        # Input: [Batch, N_Agents, N_Actions] -> Output: [Batch, N_Agents, Embed_Dim]
        self.action_encoder = nn.Linear(self.n_actions*self.n_agents, self.action_embed_dim)
        
        # 2. State Encoder: Extracts features from the global state
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 3. Fusion Layer: Combines state features and joint action embeddings
        # Input dim: State_Hidden + (N_Agents * Action_Embed)
        self.fusion_input_dim = self.hidden_dim + self.action_embed_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 2*self.hidden_dim),
            nn.ReLU()
        )
        
        # 4. Dynamics Core (GRU): Handles temporal dependencies (history/inertia)
        self.gru = nn.GRUCell(2*self.hidden_dim, self.hidden_dim)
        
        # 5. Prediction Head (Residual Head): Predicts the CHANGE in state (Delta S)
        # We predict Delta S instead of S_next directly for better stability
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim) # Output: Delta S
        )

    def forward(self, states, actions, hidden_state=None):
        """
        Forward pass for the World Model.
        
        Args:
            states:  [Batch, Time, State_Dim] - Sequence of global states
            actions: [Batch, Time, N_Agents, N_Actions] - Sequence of one-hot joint actions
            hidden_state: [Batch, Hidden_Dim] - Initial GRU hidden state (optional)
            
        Returns:
            predicted_next_states: [Batch, Time, State_Dim]
            next_hidden_state: [Batch, Hidden_Dim] (Final hidden state)
        """
        b, t, s_dim = states.size()
        
        # Container for predictions and hidden state
        predicted_next_states = []
        all_hidden_states = []
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = states.new_zeros(b, self.hidden_dim)
            
        # --- Pre-processing (Batch optimization) ---
        # 1. Encode Actions: [B, T, N, A] -> [B, T, N, Embed]

        # Flatten joint actions: [B, T, N * Embed]
        joint_act_embeds = F.relu(self.action_encoder(actions.reshape(b, t, -1)))
        
        # 2. Encode States: [B, T, S] -> [B, T, Hidden]
        state_embeds = self.state_encoder(states)
        
        # --- Temporal Loop ---
        for i in range(t):
            # Extract features for the current timestep
            curr_state_embed = state_embeds[:, i]      # [B, Hidden]
            curr_act_embed = joint_act_embeds[:, i]    # [B, N*Embed]
            
            # Fuse State and Action information
            fusion_in = torch.cat([curr_state_embed, curr_act_embed], dim=-1)
            fusion_out = self.fusion_layer(fusion_in)

            # store the hidden state
            all_hidden_states.append(hidden_state)
            
            # GRU Update: Capture history
            hidden_state = self.gru(fusion_out, hidden_state)
            
            # Predict Delta S (Change in state)
            delta_s = self.pred_head(hidden_state)
            
            # Residual Connection: S_next = S_curr + Delta_S
            # Note: We use the GROUND TRUTH S_curr from input to predict S_next (Teacher Forcing)
            next_state_pred = states[:, i] + delta_s
            
            predicted_next_states.append(next_state_pred)
            
        # Stack predictions and hidden_state back to [Batch, Time, State_Dim]
        predicted_next_states = torch.stack(predicted_next_states, dim=1)
        all_hidden_states = torch.stack(all_hidden_states, dim=1)
        
        return predicted_next_states, all_hidden_states.detach()
    
    def predict(self, states, actions, hidden_states):

        _, n_agents, n_actions = actions.shape
        joint_act_embeds = F.relu(self.action_encoder(actions.reshape(-1, n_agents*n_actions)))
        state_embeds = self.state_encoder(states)

        fusion_in = torch.cat([state_embeds, joint_act_embeds], dim=-1)
        fusion_out = self.fusion_layer(fusion_in)

        h_new = self.gru(fusion_out, hidden_states)

        delta_s = self.pred_head(h_new)
        s_next_pred = states + delta_s

        return s_next_pred
