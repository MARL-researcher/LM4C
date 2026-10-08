import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CDM(nn.Module):
    def __init__(self, args):
        super(CDM, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        # Calculate total state dimension
        self.state_dim = int(np.prod(args.state_shape))

        # Hyperparameters
        self.hidden_dim = args.rnn_hidden_dim
        self.action_embed_dim = args.action_embed_dim

        # 1. Action Encoder: Maps discrete/one-hot actions to dense embeddings
        self.action_encoder = nn.Linear(self.n_actions * self.n_agents, self.action_embed_dim)

        # 2. State Encoder: Extracts features from the global state
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 3. Fusion Layer: Combines state features and joint action embeddings
        self.fusion_input_dim = self.hidden_dim + self.action_embed_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 2 * self.hidden_dim),
            nn.ReLU()
        )

        # 4. Dynamics Core (GRU): Handles temporal dependencies (history/inertia)
        self.gru = nn.GRUCell(2 * self.hidden_dim, self.hidden_dim)

        # 5. State Prediction Head (Residual Head): Predicts Delta S
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim)
        )

        # 6. Policy Prediction Head: Predicts next-timestep Q-logits for all agents
        # Takes hidden + predicted_next_state as input
        self.policy_pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_agents * self.n_actions)
        )

    def forward(self, states, actions, hidden_state=None):
        """
        Forward pass: predicts both next state and next joint Q-logits.

        Args:
            states:  [Batch, Time, State_Dim]
            actions: [Batch, Time, N_Agents, N_Actions]
            hidden_state: [Batch, Hidden_Dim] (optional)

        Returns:
            predicted_next_states:  [Batch, Time, State_Dim]
            predicted_next_policies: [Batch, Time, N_Agents, N_Actions]
            all_hidden_states:      [Batch, Time, Hidden_Dim]
        """
        b, t, s_dim = states.size()

        predicted_next_states = []
        predicted_next_policies = []
        all_hidden_states = []

        if hidden_state is None:
            hidden_state = states.new_zeros(b, self.hidden_dim)

        # Encode Actions and States (batched over time)
        joint_act_embeds = F.relu(self.action_encoder(actions.reshape(b, t, -1)))
        state_embeds = self.state_encoder(states)

        for i in range(t):
            curr_state_embed = state_embeds[:, i]
            curr_act_embed = joint_act_embeds[:, i]

            fusion_in = torch.cat([curr_state_embed, curr_act_embed], dim=-1)
            fusion_out = self.fusion_layer(fusion_in)

            all_hidden_states.append(hidden_state)

            hidden_state = self.gru(fusion_out, hidden_state)

            # Head 1: predict Delta S
            delta_s = self.pred_head(hidden_state)
            next_state_pred = states[:, i] + delta_s

            # Head 2: predict next Q-logits (conditioned on predicted next state)
            next_policy_logits = self.policy_pred_head(hidden_state)
            next_policy_logits = next_policy_logits.reshape(b, self.n_agents, self.n_actions)

            predicted_next_states.append(next_state_pred)
            predicted_next_policies.append(next_policy_logits)

        predicted_next_states = torch.stack(predicted_next_states, dim=1)
        predicted_next_policies = torch.stack(predicted_next_policies, dim=1)
        all_hidden_states = torch.stack(all_hidden_states, dim=1)

        return predicted_next_states, predicted_next_policies, all_hidden_states.detach()

    def predict(self, states, actions, hidden_states, noise_level=0.0):
        """
        Prediction for counterfactual state rollouts.

        Args:
            states:        [*, State_Dim]
            actions:       [*, N_Agents, N_Actions]
            hidden_states: [*, Hidden_Dim]
            noise_level:   scalar

        Returns:
            s_next_pred: [*, State_Dim]
        """
        n_agents, n_actions = actions.shape[-2], actions.shape[-1]
        batch_shape = states.shape[:-1]

        joint_act_embeds = F.relu(self.action_encoder(actions.reshape(-1, n_agents * n_actions)))
        state_embeds = self.state_encoder(states.reshape(-1, self.state_dim))

        fusion_in = torch.cat([state_embeds, joint_act_embeds], dim=-1)
        fusion_out = self.fusion_layer(fusion_in)

        h_new = self.gru(fusion_out, hidden_states.reshape(-1, self.hidden_dim))

        delta_s = self.pred_head(h_new)

        if noise_level > 0.0:
            noise = torch.randn_like(delta_s)
            delta_s = delta_s + noise * (delta_s.abs() * noise_level)

        s_next_pred = states.reshape(-1, self.state_dim) + delta_s

        return s_next_pred.reshape(*batch_shape, self.state_dim)

    def predict_policy(self, states, actions, hidden_states):
        """
        Single-step prediction of next Q-logits, for I_causal computation.

        Args:
            states:        [*, State_Dim]
            actions:       [*, N_Agents, N_Actions]
            hidden_states: [*, Hidden_Dim]

        Returns:
            q_next:       [*, N_Agents, N_Actions] — raw Q-logits
            s_next:       [*, State_Dim]
            h_new:        [*, Hidden_Dim]
        """
        n_agents, n_actions = actions.shape[-2], actions.shape[-1]
        batch_shape = states.shape[:-1]
        flat_size = states.reshape(-1, self.state_dim).shape[0]

        joint_act_embeds = F.relu(self.action_encoder(actions.reshape(flat_size, n_agents * n_actions)))
        state_embeds = self.state_encoder(states.reshape(flat_size, self.state_dim))

        fusion_in = torch.cat([state_embeds, joint_act_embeds], dim=-1)
        fusion_out = self.fusion_layer(fusion_in)

        h_new = self.gru(fusion_out, hidden_states.reshape(flat_size, self.hidden_dim))

        # State prediction
        delta_s = self.pred_head(h_new)
        s_next = states.reshape(flat_size, self.state_dim) + delta_s

        # Policy prediction
        q_next = self.policy_pred_head(h_new)
        q_next = q_next.reshape(flat_size, n_agents, n_actions)

        return (
            q_next.reshape(*batch_shape, n_agents, n_actions),
            s_next.reshape(*batch_shape, self.state_dim),
            h_new.reshape(*batch_shape, self.hidden_dim),
        )
