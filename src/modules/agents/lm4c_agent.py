import torch.nn as nn
import torch.nn.functional as F
import torch as th

from modules.layers.multi_round_transformer_vae import TransformerEncoder

class LM4CAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LM4CAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.trans_aggregation = TransformerEncoder(input_dim=input_shape, dim_feedforward=args.rnn_hidden_dim*2, args=args)
        
        self.fc2 = nn.Sequential(
            nn.Linear(2*args.rnn_hidden_dim + args.consensus_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, alive_agents):
        inputs = inputs.view(-1, self.args.n_agents, self.input_shape)
        b, a, e = inputs.size()

        # consensus using transformer, n-round communication
        comm_info, latent_z = self.trans_aggregation(inputs, alive_agents)

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        consensus = latent_z.detach()
        pi_input = th.cat([comm_info, hh.view(b, a, -1), consensus], dim=-1)

        q = self.fc2(pi_input)

        return q.view(b*a, -1), hh.view(b, a, -1).detach(), latent_z
    