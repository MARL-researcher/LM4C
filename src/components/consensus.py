import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalConsensusGenerator(nn.Module):
    def __init__(self, args):
        """
        Global Generator (Training Only)
        """
        super(GlobalConsensusGenerator, self).__init__()

        self.temp = args.temp

        # --- 1. State Tower ---
        self.encoder_s = nn.Sequential(
            nn.Linear(args.state_shape, args.consensus_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.consensus_hidden_dim, args.consensus_dim),
            nn.ReLU(),
        )

        # --- 2. Reconstruction Decoder ---
        self.decoder_s = nn.Sequential(
            nn.Linear(args.consensus_dim, args.consensus_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.consensus_hidden_dim, args.state_shape),
            nn.ReLU(),
        )

        # --- 3. Context Tower ---
        input_dim_c = args.n_actions*args.n_agents + args.n_agents

        self.encoder_c = nn.Sequential(
            nn.Linear(input_dim_c, args.consensus_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.consensus_hidden_dim, args.consensus_dim),
            nn.ReLU(),
        )

    def forward(self, state, joint_action, diligence):

        # joint_action: [N, U_dim], diligence: [N, D_dim] -> [N, U+D]
        context_input = torch.cat([joint_action, diligence], dim=-1)

        z_s = self.encoder_s(state)
        z_c = self.encoder_c(context_input)

        pred_s = self.decoder_s(z_s)

        z_s_norm = F.normalize(z_s, p=2, dim=1)
        z_c_norm = F.normalize(z_c, p=2, dim=1)

        return pred_s, z_s_norm, z_c_norm
    
    def get_target_embedding(self, state):
        """
        Get target embedding for the given state
        Args:
            state: [B, T, S_dim]
        Returns:
            z_tgt: [B, T, C_dim]
        """

        z_s = self.encoder_s(state)
        z_s_norm = F.normalize(z_s, p=2, dim=1)

        return z_s_norm

class LocalConsensusGenerator(nn.Module):
    def __init__(self, args):
        super(LocalConsensusGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.state_shape),
            nn.ReLU(),
            nn.Linear(args.state_shape, args.consensus_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.consensus_hidden_dim, args.consensus_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=-1)
            
        return z