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

        # recurrent hidden + next state
        head_input_dim = 2 * self.hidden_dim

        self.policy_pred_head = nn.Sequential(
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_agents * self.n_actions),
        )

        self.context_pred_head = nn.Sequential(
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _step(self, states, actions, hidden_states):
        """
        states:        [M, S]
        actions:       [M, N, A]
        hidden_states: [M, H]
        """
        action_embed = F.relu(
            self.action_encoder(actions.reshape(-1, self.n_agents * self.n_actions))
        )
        state_embed = self.state_encoder(states)

        fusion = self.fusion_layer(
            torch.cat([state_embed, action_embed], dim=-1)
        )
        h_new = self.gru(fusion, hidden_states)

        s_next = states + self.pred_head(h_new)

        features = torch.cat(
            [h_new, self.state_encoder(s_next)], dim=-1
        )

        q_next = self.policy_pred_head(features).reshape(
            -1, self.n_agents, self.n_actions
        )
        xi_next = self.context_pred_head(features)

        return s_next, q_next, xi_next, h_new

    def forward(self, states, actions, hidden_state=None):
        b, t, _ = states.shape

        if hidden_state is None:
            hidden_state = states.new_zeros(b, self.hidden_dim)

        state_out = []
        q_out = []
        context_out = []
        hidden_out = []

        for step in range(t):
            hidden_out.append(hidden_state)

            s_next, q_next, xi_next, hidden_state = self._step(
                states[:, step],
                actions[:, step],
                hidden_state,
            )

            state_out.append(s_next)
            q_out.append(q_next)
            context_out.append(xi_next)

        return (
            torch.stack(state_out, dim=1),
            torch.stack(q_out, dim=1),
            torch.stack(context_out, dim=1),
            torch.stack(hidden_out, dim=1).detach(),
        )

    def predict(self, states, actions, hidden_states, noise_level=0.0):
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

        features = torch.cat(
            [h_new, self.state_encoder(s_next_pred)], dim=-1
        )
        q_next_pred = self.policy_pred_head(features)
        xi_next_pred = self.context_pred_head(features)

        return (
            q_next_pred.reshape(*batch_shape, self.n_agents, self.n_actions),
            s_next_pred.reshape(*batch_shape, self.state_dim),
            xi_next_pred.reshape(*batch_shape, self.hidden_dim),
        )

    def predict_policy(self, states, actions, hidden_states):
        batch_shape = states.shape[:-1]

        s_next, q_next, xi_next, _ = self._step(
            states.reshape(-1, self.state_dim),
            actions.reshape(-1, self.n_agents, self.n_actions),
            hidden_states.reshape(-1, self.hidden_dim),
        )

        return (
            q_next.reshape(*batch_shape, self.n_agents, self.n_actions),
            s_next.reshape(*batch_shape, self.state_dim),
            xi_next.reshape(*batch_shape, self.hidden_dim),
        )
