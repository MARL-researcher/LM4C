import copy

from components.episode_buffer import EpisodeBatch
from modules.mixers.gsrmixer import GSRMixer as Mixer
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets

import torch as th
from torch.optim import Adam
import torch.nn.functional as F

from components.state_predictor import ResidualWorldModel
from components.consensus import GlobalConsensusGenerator

class LM4CLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        self.mixer = Mixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

        self.weight_cs = getattr(args, "weight_cs", 0.3)

        # state predictor
        self.state_predictor = ResidualWorldModel(args)
        self.target_state_predictor = copy.deepcopy(self.state_predictor)
        self.state_predictor_params = list(self.state_predictor.parameters())
        self.state_predictor_optimiser = Adam(params=self.state_predictor_params, lr=args.predictor_lr)

        # counterfactual action
        self.cf_action = getattr(self.args, "cf_action", 0.0)

        # global consensus generator
        self.global_consensus_generator = GlobalConsensusGenerator(args)
        self.target_global_consensus_generator = copy.deepcopy(self.global_consensus_generator)
        self.global_consensus_generator_params = list(self.global_consensus_generator.parameters())
        self.global_consensus_generator_optimiser = Adam(params=self.global_consensus_generator_params, lr=args.consensus_lr)

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.args.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions = batch["actions"]
        alive_mask = batch["alive_agents"][:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # update the state predictor
        self._update_state_predictor(batch, mask)
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        trajectory_out = []
        latent_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_tragectory_outs, agent_latent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            trajectory_out.append(agent_tragectory_outs)
            latent_out.append(agent_latent_outs)
        # Concat over time
        mac_out = th.stack(mac_out, dim=1)
        trajectory_out = th.stack(trajectory_out, dim=1)
        latent_out = th.stack(latent_out, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            target_tragectory_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, target_agent_trajectory_outs, _ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                target_tragectory_out.append(target_agent_trajectory_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            # Concat across time
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_tragectory_out = th.stack(target_tragectory_out, dim=1)

            # Max over target Q-Values/ Double q learning, double Q here
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            all_target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(all_target_max_qvals, target_tragectory_out, batch["state"])

            # build counterfactual intrinsic reward for lazy agent
            mixed_next_state = self._predict_next_states(batch)
            cf_ins_rew, agent_cf_value = self._predict_cf_value(mixed_next_state, target_tragectory_out, all_target_max_qvals, mask, alive_mask)
            
            # mix the ext and ins reward
            mixed_rewards = rewards + self.args.weight_ir * cf_ins_rew

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, target_tragectory_out, batch["state"])

                targets = build_q_lambda_targets(mixed_rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(mixed_rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals, k = self.mixer(chosen_action_qvals, trajectory_out, batch["state"], output_k=True)
        chosen_action_qvals = chosen_action_qvals[:, :-1]

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        td_loss = masked_td_error.sum() / mask.sum()

        # update global VAE and construct consensus loss
        self._update_global_consensus(batch["state"][:, :-1], batch["actions_onehot"][:, :-1], agent_cf_value, mask)
        consensus_loss = self._compute_consensus_loss(latent_out[:, :-1], batch["state"][:, :-1], mask)

        loss = td_loss + self.weight_cs*consensus_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            self.logger.log_stat("reward", (rewards * mask).sum() / (mask.sum() + 1e-8), t_env)
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("loss_consensus", consensus_loss.item(), t_env)
            self.logger.log_stat("total_loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info
    
    def _update_state_predictor(self, batch, mask):

        state_in = batch["state"][:, :-1]
        action_in = batch["actions_onehot"][:, :-1]

        state_target = batch["state"][:, 1:]

        pred_state_out, _ = self.state_predictor(state_in, action_in)

        wm_loss = F.mse_loss(pred_state_out, state_target, reduction='none').mean(dim=-1, keepdim=True)
        wm_loss = (wm_loss * mask).sum() / (mask.sum() + 1e-8)

        self.state_predictor_optimiser.zero_grad()
        wm_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.state_predictor_params, self.args.grad_norm_clip)
        self.state_predictor_optimiser.step()


    def _predict_cf_value(self, mixed_state, trajectory, qval, seq_mask, alive_mask):

        # real V
        s_next_real = mixed_state[:, :, 0, 0, :]    # [B, T, S], tensor are the same in dim N
        v_real = self.target_mixer(qval[:, :-1], trajectory[:, :-1], s_next_real)   # [B, T, 1]

        # lazy V
        bs, seq_len, n_agents, _, s_dim = mixed_state.shape
        s_next_lazy = mixed_state[:, :, :, 1, :]
        s_next_lazy = s_next_lazy.reshape(bs, seq_len*n_agents, s_dim)
        exp_trajectory = (trajectory[:, :-1]).unsqueeze(2).expand(-1, -1, n_agents, -1, -1)
        exp_trajectory = exp_trajectory.reshape(bs, seq_len*n_agents, n_agents, -1)
        exp_qval = (qval[:, :-1]).unsqueeze(2).expand(-1, -1, n_agents, -1)
        exp_qval = exp_qval.reshape(bs, seq_len*n_agents, -1)
        v_lazy = self.target_mixer(exp_qval, exp_trajectory, s_next_lazy)
        v_lazy = v_lazy.reshape(bs, seq_len, n_agents, -1)

        # lazy value for all agents
        v_delta = v_real.unsqueeze(2) - v_lazy
        # scale the intrinsic reward
        v_delta = (F.relu(v_delta)).squeeze()

        total_mask = seq_mask * alive_mask.squeeze()
        agent_cf_value = v_delta * total_mask
        cf_value = agent_cf_value.sum(dim=-1) / (total_mask.sum(dim=-1) + 1e-8)
        
        return cf_value.unsqueeze(-1), agent_cf_value

    def _predict_next_states(self, batch):
        """
        Predict both REAL and LAZY next states simultaneously in a paired structure.
        
        Logic: Transform inputs to [Batch, Time, N_Agents, 2, ...] and infer once.
        - Index 0: Original Joint Action (Real)
        - Index 1: Joint Action where 'Target Agent' is set to Stay (Lazy)
        
        Args:
            batch: PyMARL batch object
            wm_hidden_state: [Batch, Time, Hidden_Dim] (Context before update)
            
        Returns:
            mixed_next_states: [Batch, Time, N_Agents, 2, State_Dim]
        """
        # Data Slicing (Time 0 to T-1)
        s_curr = batch["state"][:, :-1]
        u_curr = batch["actions_onehot"][:, :-1]

        _, wm_hidden_state = self.target_state_predictor(s_curr, u_curr)

        # Ensure wm_hidden_state matches the sequence length of inputs
        if wm_hidden_state.shape[1] > s_curr.shape[1]:
            h_ctx = wm_hidden_state[:, :s_curr.shape[1]]
        else:
            h_ctx = wm_hidden_state
        
        bs, seq_len, n_agents, n_actions = u_curr.shape
        hidden_dim, state_dim = h_ctx.shape[-1], s_curr.shape[-1]

        # s_curr: [B, T, S] -> [B, T, 1, 1, S] -> [B, T, N, 2, S]
        s_pair = s_curr.unsqueeze(2).unsqueeze(3).expand(-1, -1, n_agents, 2, -1)
        # h_ctx: [B, T, H] -> [B, T, 1, 1, H] -> [B, T, N, 2, H]
        h_pair = h_ctx.unsqueeze(2).unsqueeze(3).expand(-1, -1, n_agents, 2, -1)

        # u_curr: [B, T, N, A] -> [B, T, 1, 1, N, A] -> [B, T, N, 2, N, A]
        u_pair = u_curr.unsqueeze(2).unsqueeze(3).expand(-1, -1, n_agents, 2, -1, -1).clone()

        # apply mask to the lazy half
        target_agent_ids = th.arange(n_agents, device=self.device)
        cf_action = th.zeros(self.args.n_actions, device=self.device)
        cf_action[self.cf_action] = 1.0
        u_pair[:, :, target_agent_ids, 1, target_agent_ids, :] = cf_action

        s_flat = s_pair.reshape(-1, state_dim)
        h_flat = h_pair.reshape(-1, hidden_dim)
        u_flat = u_pair.reshape(-1, n_agents, n_actions)

        # use the target predictor
        mixed_next_state = self.target_state_predictor.predict(s_flat, u_flat, h_flat)
        mixed_next_state = mixed_next_state.view(bs, seq_len, n_agents, 2, state_dim)

        return mixed_next_state
    
    def _compute_consensus_loss(self, pred_z, true_state, mask):
        
        with th.no_grad():
            target_z = self.target_global_consensus_generator.get_target_embedding(true_state)
            target_z = target_z.unsqueeze(dim=2).expand(-1, -1, pred_z.shape[2], -1)
        
        student_loss = th.mean(F.mse_loss(pred_z, target_z, reduction='none'), dim=-1)
        student_loss = th.mean(student_loss, dim=-1, keepdim=True)
        student_loss = (student_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return student_loss
    
    def _update_global_consensus(self, state, action, cf_value, mask, weight_nce=0.1):

        batch_size, t_horizon, _ = state.shape
        flat_size = batch_size * t_horizon

        state_flat = state.reshape(flat_size, -1)
        diligence_flat = cf_value.reshape(flat_size, -1)
        action_flat = (action.reshape(batch_size, t_horizon, -1)).reshape(flat_size, -1)
        mask_flat = mask.reshape(flat_size, -1)

        # protect for the GPU memory
        if flat_size > self.args.max_sample_size:
            indices = th.randperm(flat_size)[:self.args.max_sample_size]
            s_batch, u_batch, d_batch = state_flat[indices], action_flat[indices], diligence_flat[indices]
            mask_batch = mask_flat[indices]
        else:
            s_batch, u_batch, d_batch = state_flat, action_flat, diligence_flat
            mask_batch = mask_flat

        recon_s, z_s_norm, z_ctx_norm = self.global_consensus_generator(s_batch, u_batch, d_batch)

        # loss 1: InfoNCE
        logits = th.matmul(z_s_norm, z_ctx_norm.T) / self.args.temp
        labels = th.arange(logits.shape[0]).to(self.device)
        loss_nce = (F.cross_entropy(logits, labels, reduction='none')).unsqueeze(dim=-1)
        loss_nce = (loss_nce * mask_batch).sum() / (mask_batch.sum() + 1e-8)

        # loss 2: reconstruction loss
        loss_recon = th.mean(F.mse_loss(recon_s, s_batch, reduction='none'), dim=-1, keepdim=True)
        loss_recon = (loss_recon * mask_batch).sum() / (mask_batch.sum() + 1e-8)

        # total loss
        loss_teach = loss_recon + weight_nce*loss_nce

        # optimize
        self.global_consensus_generator_optimiser.zero_grad()
        loss_teach.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.global_consensus_generator_params, self.args.grad_norm_clip)
        self.global_consensus_generator_optimiser.step()
        
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_state_predictor.load_state_dict(self.state_predictor.state_dict())
        self.target_global_consensus_generator.load_state_dict(self.global_consensus_generator.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.state_predictor.cuda()
        self.target_state_predictor.cuda()
        self.global_consensus_generator.cuda()
        self.target_global_consensus_generator.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/policy_opt.th".format(path))

        th.save(self.target_state_predictor.state_dict(), "{}/target_state_predictor.th".format(path))
        th.save(self.state_predictor.state_dict(), "{}/state_predictor.th".format(path))
        th.save(self.state_predictor_optimiser.state_dict(), "{}/state_predictor_opt.th".format(path))

        th.save(self.global_consensus_generator.state_dict(), "{}/global_consensus_generator.th".format(path))
        th.save(self.target_global_consensus_generator.state_dict(), "{}/target_global_consensus_generator.th".format(path))
        th.save(self.global_consensus_generator_optimiser.state_dict(), "{}/global_consensus_generator_optimiser.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

        self.optimiser.load_state_dict(th.load("{}/policy_opt.th".format(path), map_location=lambda storage, loc: storage))

        self.state_predictor.load_state_dict(th.load("{}/state_predictor.th".format(path), map_location=lambda storage, loc: storage))
        self.target_state_predictor.load_state_dict(th.load("{}/state_predictor.th".format(path), map_location=lambda storage, loc: storage))
        self.state_predictor_optimiser.load_state_dict(th.load("{}/state_predictor_opt.th".format(path), map_location=lambda storage, loc: storage))

        self.global_consensus_generator.load_state_dict(th.load("{}/global_consensus_generator.th".format(path), map_location=lambda storage, loc: storage))
        self.target_global_consensus_generator.load_state_dict(th.load("{}/target_global_consensus_generator.th".format(path), map_location=lambda storage, loc: storage))
        self.global_consensus_generator_optimiser.load_state_dict(th.load("{}/global_consensus_generator_optimiser.th".format(path), map_location=lambda storage, loc: storage))