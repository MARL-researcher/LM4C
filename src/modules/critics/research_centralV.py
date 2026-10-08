import torch as th
import torch.nn as nn
import torch.nn.functional as F
from components.soft_attention import SoftAttention

class ResearchCentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ResearchCentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

        self.att_state_layer = SoftAttention(args)

    def forward(self, batch, tragectory, t=None, importance=False, scale_factor=100):
        if importance is False:
            inputs, bs, max_t = self._build_inputs(batch, tragectory, t=t)
            x = F.relu(self.fc1(inputs))
            x = F.relu(self.fc2(x))
            q = self.fc3(x)
            return q
        else:
            inputs, masked_inputs, bs, max_t = self._build_inputs_with_importance(batch, tragectory, t=t)
            x = F.relu(self.fc1(inputs))
            x = F.relu(self.fc2(x))
            q = self.fc3(x)

            x_i= F.relu(self.fc1(masked_inputs))
            x_i = F.relu(self.fc2(x_i))
            q_i = self.fc3(x_i)

            delta_imp = q - q_i
            delta_imp = F.sigmoid(delta_imp.abs()*scale_factor)*2 - 1

            return q, delta_imp.squeeze(-1).detach()

    def _build_inputs(self, batch, tragectory, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        att_states = self.att_state_layer(tragectory, batch["state"])
        inputs.append(att_states[:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t
    
    def _build_inputs_with_importance(self, batch, tragectory, t=None):
        """
        Build normal inputs and masked inputs for importance calculation.
        The masked version zeros each agent's own action (not the state).
        """

        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # --- Step 1: Compute the attended state  ---
        att_states = self.att_state_layer(tragectory, batch["state"])
        inputs.append(att_states[:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # --- Step 2: Prepare action input ---
        if self.args.obs_last_action:
            if t == 0:
                actions = th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1)
            elif isinstance(t, int):
                actions = batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1)
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(actions)

        # --- Step 3: Agent identity matrix ---
        agent_id_matrix = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        inputs.append(agent_id_matrix)

        # --- Step 4: Build full inputs ---
        inputs = th.cat(inputs, dim=-1)

        # TODO: Verify that this implementation is correct.
        # --- Step 5: Mask each agent’s action (zero out its own one-hot) ---
        masked_inputs = []
        if self.args.mask_state:
            masked_att_states = self.att_state_layer(tragectory, batch["state"], mask=True)
            masked_inputs.append(masked_att_states)
        else:
            masked_inputs.append(att_states[:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        if self.args.obs_last_action:
            actions = actions.view(bs, max_t, self.n_agents, self.n_agents, -1)
            mask = (1 - th.eye(self.n_agents, device=batch.device)).view(1, 1, self.n_agents, self.n_agents, 1)
            masked_actions = actions * mask
            masked_actions = masked_actions.view(bs, max_t, self.n_agents, -1)
            masked_inputs.append(masked_actions)
        
        masked_inputs.append(agent_id_matrix)
        masked_inputs = th.cat(masked_inputs, dim=-1)

        return inputs, masked_inputs, bs, max_t

    def _min_max_normalize(self, tensor):
        min = tensor.min()
        max = tensor.max()
        return (tensor - min) / (max - min + 1e-8)

    def _get_input_shape(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape