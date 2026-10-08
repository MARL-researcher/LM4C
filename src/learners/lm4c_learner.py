import copy

from components.episode_buffer import EpisodeBatch
from modules.mixers.research_mixer_aamas import ResearchMixer as Mixer
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets

import torch as th
from torch.optim import Adam
import torch.nn.functional as F

from components.casual_dynamics_model import CDM
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
        self.target_mixer.requires_grad_(False)
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
        self.policy_predictor = CDM(args)
        self.target_state_predictor = copy.deepcopy(self.policy_predictor)
        self.state_predictor_params = list(self.policy_predictor.parameters())
        self.cdm_optimiser = Adam(params=self.state_predictor_params, lr=args.predictor_lr)

        self.policy_loss_weight = getattr(self.args, "weight_Q", 1.0)

        self.ctx_loss_weight = getattr(self.args, "weight_ctx", 1.0)
        self.val_loss_weight = getattr(self.args, "weight_val", 1.0)

        self.nce_loss_weight = getattr(self.args, "weight_nce", 0.1)

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
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        
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

        # update the policy predictor
        joint_policy_out = mac_out.clone().detach()

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

        state_wm_log = self._update_policy_predictor(batch, target_mac_out, target_tragectory_out, mask)

        with th.no_grad():
            # Max over target Q-Values/ Double q learning, double Q here
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            all_target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(all_target_max_qvals, target_tragectory_out, batch["state"])

            is_smac = self.args.env in ("sc2_v1", "sc2_v2")
            compute_fn = self._compute_causal_influence_per_agent if is_smac else self._compute_causal_influence
            I_causal, A_fv = compute_fn(batch, joint_policy_out, mask)

            # compute diligence
            diligence = I_causal * F.relu(A_fv)
            diligence_rew = diligence.sum(dim=-1, keepdim=True)

            # mix the ext and ins reward (use diligence as intrinsic reward)
            mixed_rewards = rewards + self.args.weight_ir * diligence_rew

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
        consensus_running_log = self._update_global_consensus(batch["state"][:, :-1], batch["actions_onehot"][:, :-1], diligence, mask, self.nce_loss_weight)
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

            for key in consensus_running_log:
                self.logger.log_stat(key, consensus_running_log[key][0], t_env)

            for key in state_wm_log:
                self.logger.log_stat(key, state_wm_log[key][0], t_env)

            self.logger.log_stat("external_reward", (rewards * mask).sum() / (mask.sum() + 1e-8), t_env)
            mean_diligence = (diligence_rew * mask).sum() / (mask.sum() + 1e-8)
            mean_I_causal = (I_causal.mean(dim=-1, keepdim=True) * mask).sum() / (mask.sum() + 1e-8)
            mean_A_fv = (A_fv.mean(dim=-1, keepdim=True) * mask).sum() / (mask.sum() + 1e-8)
            self.logger.log_stat("diligence_reward", mean_diligence, t_env)
            self.logger.log_stat("weighted_diligence_reward", mean_diligence*self.args.weight_ir, t_env)
            self.logger.log_stat("I_causal_mean", mean_I_causal, t_env)
            self.logger.log_stat("A_fv_mean", mean_A_fv, t_env)
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("weighted_loss_consensus", consensus_loss.item()*self.weight_cs, t_env)
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
    
    def _update_policy_predictor(
    self, batch, target_mac_out, target_trajectory_out, mask,
):
        state_in = batch["state"][:, :-1]
        action_in = batch["actions_onehot"][:, :-1]

        with th.no_grad():
            state_target = batch["state"][:, 1:]
            q_target = target_mac_out[:, 1:].detach()

            ctx_target = self.target_mixer.encode_context(
                target_trajectory_out[:, 1:],
                state_target,
            )

            value_target = self.target_mixer.mix_core(
                self._rho(q_target),
                ctx_target,
            )

        pred_state, pred_q, pred_ctx, _ = self.policy_predictor(
            state_in, action_in
        )

        # Keep this outside no_grad; pred_q and pred_ctx must also remain attached to the computation graph.
        pred_value = self.target_mixer.mix_core(
            self._rho(pred_q),
            pred_ctx,
        )

        def masked_mse(pred, target):
            # Average each term over the feature dimension to obtain [B, T-1, 1].
            error = (pred - target).pow(2)
            error = error.reshape(
                mask.shape[0], mask.shape[1], -1
            ).mean(dim=-1, keepdim=True)

            return (error * mask).sum() / (mask.sum() + 1e-8)

        state_loss = masked_mse(pred_state, state_target)
        q_loss = masked_mse(pred_q, q_target)
        ctx_loss = masked_mse(pred_ctx, ctx_target)
        val_loss = masked_mse(pred_value, value_target)

        total_loss = (
            state_loss
            + self.policy_loss_weight * q_loss
            + self.ctx_loss_weight * ctx_loss
            + self.val_loss_weight * val_loss
        )

        self.cdm_optimiser.zero_grad()
        total_loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(
            self.state_predictor_params,
            self.args.grad_norm_clip,
        )
        self.cdm_optimiser.step()

        return {
            "cdm_state_loss": [state_loss.item()],
            "cdm_q_loss": [q_loss.item()],
            "cdm_ctx_loss": [ctx_loss.item()],
            "cdm_val_loss": [val_loss.item()],
            "cdm_total_loss": [total_loss.item()],
            "cdm_grad_norm": [grad_norm.item()],
        }

    @th.no_grad()
    def _compute_causal_influence(self, batch, mac_out, mask):
        s_curr = batch["state"][:, :-1]            # [B, T-1, S]
        u_curr = batch["actions_onehot"][:, :-1]   # [B, T-1, N, A]
        avail = batch["avail_actions"][:, :-1]      # [B, T-1, N, A]

        bs, seq, n_agents, n_actions = u_curr.shape
        flat_size = bs * seq
        device = s_curr.device

        # π_i: agent policy from Q-values at time t (marginalization weights)
        pi = self._masked_policy(mac_out[:, :-1], avail)

        # Hidden states from target CDM
        _, _, _, cdm_hidden = self.target_state_predictor(s_curr, u_curr)
        # cdm_hidden: [B, T-1, H]

        # --- Flatten all inputs ---
        s_flat = s_curr.reshape(flat_size, -1)               # [B*T, S]
        u_flat = u_curr.reshape(flat_size, n_agents, n_actions)  # [B*T, N, A]
        h_flat = cdm_hidden.reshape(flat_size, -1)            # [B*T, H]

        # --- q̂_do, s_do: CDM with real joint action ---
        q_do, _, _ = self.target_state_predictor.predict_policy(s_flat, u_flat, h_flat)
        q_do = q_do.reshape(bs, seq, n_agents, n_actions)   # [B, T-1, N, A]
        # s_do: [B*T, S] — predicted next state under the actual joint action

        # --- Build all counterfactuals: [N, A, B*T, N, A] ---
        N, A = n_agents, n_actions

        # u_cf[i, a, k, :, :] = u_flat[k, :, :] with agent i's action replaced by one_hot(a)
        u_cf = u_flat[None, None, :, :, :].expand(N, A, -1, -1, -1).clone()

        # Replace agent i's action with one_hot(a): u_cf[i, a, :, i, :] = eye[a]
        eye = th.eye(A, device=device)  # [A, A]
        for i in range(N):
            u_cf[i, :, :, i, :] = eye.unsqueeze(1).expand(A, flat_size, A)

        # Flatten: [N, A, B*T, N, A] → [N*A*B*T, N, A]
        u_cf_all = u_cf.reshape(N * A * flat_size, N, A)
        s_cf_all = s_flat[None, None, :, :].expand(N, A, -1, -1).reshape(N * A * flat_size, -1)
        h_cf_all = h_flat[None, None, :, :].expand(N, A, -1, -1).reshape(N * A * flat_size, -1)

        # --- One batched forward for ALL counterfactuals (Q-logits + states) ---
        q_cf_all, _, xi_cf_all = self.target_state_predictor.predict_policy(s_cf_all, u_cf_all, h_cf_all)
        # q_cf_all: [N*A*B*T, N, A] → [N, A, B*T, N, A]
        q_cf = q_cf_all.reshape(N, A, flat_size, N, A)

        # ==================== I_causal ====================
        # Marginalization weights under the current policy.
        pi_flat = pi.reshape(flat_size, N, A)
        weight = pi_flat.permute(1, 2, 0)  # [N, A, flat_size]

        # Compute the weighted Q-response over all candidate actions for each intervened agent.
        q_marg = (
            q_cf * weight[:, :, :, None, None]
        ).sum(dim=1)  # [N, flat_size, N, A]

        q_do_flat = q_do.reshape(flat_size, N, A)

        # Difference between the actual-action response and the marginalized response.
        diff = q_do_flat.unsqueeze(0) - q_marg

        # Exclude the intervened agent's own Q-response.
        mask_self = 1.0 - th.eye(
            N, device=device, dtype=q_cf.dtype
        )
        diff = diff * mask_self[:, None, :, None]

        I_causal_flat = diff.square().sum(
            dim=(-2, -1)
        )  # [N, flat_size]

        # Each action branch uses its own predicted Q and context.
        value_cf = self.target_mixer.mix_core(self._rho(q_cf_all).unsqueeze(1), xi_cf_all.unsqueeze(1), ).reshape(N, A, flat_size)  # [intervened agent, candidate action, sample]

        # weight has shape [N, A, flat_size] and was computed above for I_causal.
        value_bar = (value_cf * weight).sum(dim=1)  # [N, flat_size]

        # Select the branch value corresponding to each agent's executed action.
        executed_actions = (batch["actions"][:, :-1].squeeze(-1).reshape(flat_size, N).transpose(0, 1).long())  # [N, flat_size]

        value_exec = value_cf.gather(dim=1,index=executed_actions.unsqueeze(1)).squeeze(1)  # [N, flat_size]

        A_fv_flat = value_exec - value_bar

        # ==================== Reshape and mask ====================

        I_causal = I_causal_flat.permute(1, 0).reshape(bs, seq, N)
        A_fv = A_fv_flat.permute(1, 0).reshape(bs, seq, N)

        mask_exp = mask.expand(-1, -1, N)

        return I_causal * mask_exp, A_fv * mask_exp

    def _rho(self, q):
        return q.max(dim=-1).values

    def _masked_policy(self, q, avail):
        available = avail != 0
        logits = q.masked_fill(~available, -th.inf)
        # Padding may have no available actions; avoid softmax over all -inf.
        logits = th.where(
            available.any(dim=-1, keepdim=True),
            logits,
            th.zeros_like(logits),
        )
        return F.softmax(logits, dim=-1).masked_fill(~available, 0.0)

    @th.no_grad()
    def _compute_causal_influence_per_agent(self, batch, mac_out, mask):
        """
        Memory-efficient variant: processes one agent at a time to avoid
        building the full [N, A, B*T, N, A] tensor. Used when the environment
        has long episodes or many agents (e.g., SMAC).

        Same inputs and outputs as _compute_causal_influence.
        """
        s_curr = batch["state"][:, :-1]
        u_curr = batch["actions_onehot"][:, :-1]
        avail = batch["avail_actions"][:, :-1]

        bs, seq, n_agents, n_actions = u_curr.shape
        flat_size = bs * seq
        device = s_curr.device

        # π_i: agent policy from Q-values at time t (marginalization weights)
        pi = self._masked_policy(mac_out[:, :-1], avail)
        pi_flat = pi.reshape(flat_size, n_agents, n_actions)  # [B*T, N, A]

        # Hidden states from target CDM
        _, _, _, cdm_hidden = self.target_state_predictor(s_curr, u_curr)

        with th.no_grad():
            # --- Flatten all inputs ---
            s_flat = s_curr.reshape(flat_size, -1)
            u_flat = u_curr.reshape(flat_size, n_agents, n_actions)
            h_flat = cdm_hidden.reshape(flat_size, -1)

            # --- q̂_do, s_do: CDM with real joint action ---
            q_do, _, _ = self.target_state_predictor.predict_policy(s_flat, u_flat, h_flat)
            q_do = q_do.reshape(flat_size, n_agents, n_actions)

            A_fv_flat = th.zeros(n_agents, flat_size, device=device)

            executed_actions = (batch["actions"][:, :-1].squeeze(-1).reshape(flat_size, n_agents).long())

            I_causal_flat = th.zeros(n_agents, flat_size, device=device)
            A_td_flat = th.zeros(n_agents, flat_size, device=device)
            eye = th.eye(n_actions, device=device)

            # -- self-mask for I_causal: exclude agent i's own logits --
            mask_self = th.ones(n_agents, n_agents, device=device)
            for i in range(n_agents):
                mask_self[i, i] = 0.0

            for i in range(n_agents):
                # --- Build counterfactuals for agent i: [A, B*T, N, A] ---
                u_cf_i = u_flat[None, :, :, :].expand(n_actions, -1, -1, -1).clone()
                u_cf_i[:, :, i, :] = eye.unsqueeze(1).expand(n_actions, flat_size, n_actions)

                u_cf_i_flat = u_cf_i.reshape(n_actions * flat_size, n_agents, n_actions)
                s_cf_i_flat = s_flat[None, :, :].expand(n_actions, -1, -1).reshape(n_actions * flat_size, -1)
                h_cf_i_flat = h_flat[None, :, :].expand(n_actions, -1, -1).reshape(n_actions * flat_size, -1)

                # --- CDM forward for agent i ---
                q_cf_i, _, xi_cf_i = self.target_state_predictor.predict_policy(
                    s_cf_i_flat,
                    u_cf_i_flat,
                    h_cf_i_flat,
                )

                q_cf_i = q_cf_i.reshape(
                    n_actions, flat_size, n_agents, n_actions
                )

                # --- I_causal for agent i ---
                pi_i = pi_flat[:, i, :]  # [B*T, A]
                q_marg_i = (q_cf_i * pi_i.T.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)  # [B*T, N, A]

                # diff[k, j, :] = q_do[k,j,:] - q_marg_i[k,j,:], exclude agent i (j == i)
                diff_i = (q_do - q_marg_i) * mask_self[i][None, :, None]  # [B*T, N, A]
                I_causal_flat[i] = (diff_i ** 2).sum(dim=[-2, -1])  # [B*T]

                value_cf_i = self.target_mixer.mix_core(
                    self._rho(q_cf_i).reshape(n_actions * flat_size, 1, n_agents), 
                    xi_cf_i.reshape(n_actions * flat_size, 1, self.args.rnn_hidden_dim)
                    ).reshape(n_actions, flat_size)

                value_bar_i = (value_cf_i * pi_i.T).sum(dim=0)
                value_exec_i = value_cf_i.gather(dim=0, index=executed_actions[:, i].unsqueeze(0),).squeeze(0)

                A_fv_flat[i] = value_exec_i - value_bar_i

            # ==================== Reshape and mask ====================

            I_causal = I_causal_flat.permute(1, 0).reshape(bs, seq, n_agents)
            A_fv = A_fv_flat.permute(1, 0).reshape(bs, seq, n_agents)

            mask_exp = mask.expand(-1, -1, n_agents)

            return I_causal * mask_exp, A_fv * mask_exp

    def _compute_consensus_loss(self, pred_z, true_state, mask):
        
        with th.no_grad():
            target_z = self.target_global_consensus_generator.get_target_embedding(true_state)
            target_z = target_z.unsqueeze(dim=2).expand(-1, -1, pred_z.shape[2], -1)
        
        student_loss = th.mean(F.mse_loss(pred_z, target_z, reduction='none'), dim=-1)
        student_loss = th.mean(student_loss, dim=-1, keepdim=True)
        student_loss = (student_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return student_loss
    
    def _update_global_consensus(self, state, action, cf_value, mask, weight_nce):

        running_log = {
            "teach_state_recon_loss": [],
            "teach_nce_loss": [],
            "teach_weighted_nce_Loss": [],
            "teach_total_loss": []
        }

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

        running_log['teach_state_recon_loss'].append(loss_recon.item())
        running_log['teach_nce_loss'].append(loss_nce.item())
        running_log['teach_weighted_nce_Loss'].append((weight_nce * loss_nce).item())
        running_log['teach_total_loss'].append(loss_teach.item())

        return running_log
        
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_state_predictor.load_state_dict(self.policy_predictor.state_dict())
        self.target_global_consensus_generator.load_state_dict(self.global_consensus_generator.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.policy_predictor.cuda()
        self.target_state_predictor.cuda()
        self.global_consensus_generator.cuda()
        self.target_global_consensus_generator.cuda()
    
    def cal_true_latent_z(self, batch: EpisodeBatch):
        with th.no_grad():
            target_z = self.target_global_consensus_generator.get_target_embedding(batch["state"][:, :-1])
            return target_z
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/policy_opt.th".format(path))

        th.save(self.target_state_predictor.state_dict(), "{}/target_state_predictor.th".format(path))
        th.save(self.policy_predictor.state_dict(), "{}/state_predictor.th".format(path))
        th.save(self.cdm_optimiser.state_dict(), "{}/state_predictor_opt.th".format(path))

        th.save(self.global_consensus_generator.state_dict(), "{}/global_consensus_generator.th".format(path))
        th.save(self.target_global_consensus_generator.state_dict(), "{}/target_global_consensus_generator.th".format(path))
        th.save(self.global_consensus_generator_optimiser.state_dict(), "{}/global_consensus_generator_optimiser.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimiser.load_state_dict(th.load("{}/policy_opt.th".format(path), map_location=lambda storage, loc: storage))

        self.policy_predictor.load_state_dict(th.load("{}/state_predictor.th".format(path), map_location=lambda storage, loc: storage))
        self.target_state_predictor.load_state_dict(th.load("{}/state_predictor.th".format(path), map_location=lambda storage, loc: storage))
        self.cdm_optimiser.load_state_dict(th.load("{}/state_predictor_opt.th".format(path), map_location=lambda storage, loc: storage))

        self.global_consensus_generator.load_state_dict(th.load("{}/global_consensus_generator.th".format(path), map_location=lambda storage, loc: storage))
        self.target_global_consensus_generator.load_state_dict(th.load("{}/target_global_consensus_generator.th".format(path), map_location=lambda storage, loc: storage))
        self.global_consensus_generator_optimiser.load_state_dict(th.load("{}/global_consensus_generator_optimiser.th".format(path), map_location=lambda storage, loc: storage))
