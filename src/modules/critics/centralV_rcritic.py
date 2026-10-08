import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RCentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(RCentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        self._use_orthogonal = getattr(args, "use_orthogonal", False)
        self._use_layer_norm = getattr(args, "use_layer_norm", False)

        # 1. Define the network layers.
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        
        if self._use_layer_norm:
            self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)

        if args.critic_rnn:
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        else:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

        # 2. Apply orthogonal initialization.
        if self._use_orthogonal:
            # FC1 -> ReLU
            nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.fc1.bias, 0)

            # RNN
            if args.critic_rnn:
                for name, param in self.rnn.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
            else:
                # Intermediate layers in MLP mode.
                nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(self.fc2.bias, 0)
            
            # Value output usually uses gain=1.0, as in light_mappo.
            # Alternatively, use 0.01 to keep the initial value predictions close to zero.
            nn.init.orthogonal_(self.fc3.weight, gain=1.0) 
            nn.init.constant_(self.fc3.bias, 0)

    def init_hidden(self, batch_size):
        return self.fc1.weight.new(batch_size, self.n_agents, self.args.rnn_hidden_dim).zero_()

    def forward(self, batch, t=None, hidden_state=None):
        inputs, bs, _ = self._build_inputs(batch, t=t)
        
        x = F.relu(self.fc1(inputs))
        x = x.reshape(-1, self.args.rnn_hidden_dim)

        if self.args.critic_rnn:
            if hidden_state is not None:
                h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
                h = self.rnn(x, h_in)
                
                # Apply LayerNorm if enabled
                h_feat = self.layer_norm(h) if self._use_layer_norm else h
                
                q = self.fc3(h_feat)
                q = q.reshape(bs, self.n_agents, 1)
                h = h.reshape(bs, self.n_agents, self.args.rnn_hidden_dim)
                return q, h
            else:
                # This branch is generally discouraged when the first step has no hidden state, but is retained for compatibility.
                # Initialize a zero hidden state in this case.
                h_in = x.new_zeros(x.shape)
                h = self.rnn(x, h_in)
                h_feat = self.layer_norm(h) if self._use_layer_norm else h
                q = self.fc3(h_feat)
                q = q.reshape(bs, self.n_agents, 1)
                return q
        else:
            # MLP Mode
            h = F.relu(self.fc2(x))
            h_feat = self.layer_norm(h) if self._use_layer_norm else h
            q = self.fc3(h_feat)
            q = q.reshape(bs, self.n_agents, 1)
            return q
    
    def forward_step(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        x = x.reshape(-1, self.args.rnn_hidden_dim)
        
        if self.args.critic_rnn:
            if hidden_state is not None:
                h = self.rnn(x, hidden_state)
            else:
                h = x.new_zeros(x.shape) # Fallback

            h_feat = self.layer_norm(h) if self._use_layer_norm else h
            q = self.fc3(h_feat)
            return q, h
        else:
            # MLP Mode
            h = F.relu(self.fc2(x))
            h_feat = self.layer_norm(h) if self._use_layer_norm else h
            q = self.fc3(h_feat)
            return q, None

    def _build_inputs(self, batch, t=None):
        # ... (Keep the existing logic unchanged.) ...
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # ... (Keep the existing logic unchanged.) ...
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        input_shape += self.n_agents
        return input_shape