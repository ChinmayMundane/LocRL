"""Scalar SAC updates conditioned on the same preferences as CAPQL."""
from pathlib import Path
import sys
from itertools import chain

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "capql"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch
from torch import nn
from torch.nn import functional as F
from capql.train import (
    LocalizationCAPQL, ObservationEncoder, make_mlp, build_agent,
)


class ScalarQNet(nn.Module):
    """A single scalar return, conditioned on observation, action, and weights."""
    def __init__(self, image_shape, info_dim, action_dim, net_arch):
        super().__init__()
        self.encoder = ObservationEncoder(image_shape, info_dim)
        self.head = make_mlp(self.encoder.output_dim + action_dim + 2, 1, net_arch)

    def forward(self, observation, action, weights):
        return self.head(torch.cat((self.encoder(observation), action, weights), dim=-1))


def scalar_bellman_target(rewards, weights, dones, next_q, log_pi, gamma, alpha):
    """Return a [batch, 1] target; every transition retains its own preference."""
    reward = (rewards * weights).sum(dim=-1, keepdim=True)
    return reward + gamma * (1 - dones.reshape(-1, 1)) * (
        next_q.reshape(-1, 1) - alpha * log_pi.reshape(-1, 1)
    )


class PreferenceConditionedSAC(LocalizationCAPQL):
    """Reuse collection/evaluation only; all Bellman and actor updates are scalar SAC."""
    method_label = "PC-SAC"
    checkpoint_prefix = "pc_sac"

    def __init__(self, env, image_shape, info_dim, **kwargs):
        super().__init__(env, image_shape, info_dim, **kwargs)
        self.q_nets = [ScalarQNet(image_shape, info_dim, self.action_dim, self.net_arch).to(self.device)
                       for _ in range(self.num_q_nets)]
        self.target_q_nets = [ScalarQNet(image_shape, info_dim, self.action_dim, self.net_arch).to(self.device)
                              for _ in range(self.num_q_nets)]
        for critic, target in zip(self.q_nets, self.target_q_nets):
            target.load_state_dict(critic.state_dict())
            target.requires_grad_(False)
        self.q_optim = torch.optim.Adam(chain.from_iterable(q.parameters() for q in self.q_nets),
                                        lr=self.learning_rate)

    def update(self):
        for _ in range(self.gradient_updates):
            obs, actions, weights, rewards, next_obs, dones = self._sample_batch_experiences()
            with torch.no_grad():
                next_actions, log_pi, _ = self.policy.sample(next_obs, weights)
                next_q = torch.stack([q(next_obs, next_actions, weights)
                                      for q in self.target_q_nets]).min(dim=0).values
                target = scalar_bellman_target(rewards, weights, dones, next_q, log_pi,
                                               self.gamma, self.alpha)
            critic_loss = sum(F.mse_loss(q(obs, actions, weights), target)
                              for q in self.q_nets) / self.num_q_nets
            self.q_optim.zero_grad()
            critic_loss.backward()
            self.q_optim.step()

            # Keep action gradients through Q while avoiding unused critic gradients.
            for q in self.q_nets:
                q.requires_grad_(False)
            pi, log_pi, _ = self.policy.sample(obs, weights)
            q_pi = torch.stack([q(obs, pi, weights) for q in self.q_nets]).min(dim=0).values
            actor_loss = (self.alpha * log_pi.reshape(-1, 1) - q_pi).mean()
            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()
            for q in self.q_nets:
                q.requires_grad_(True)
            with torch.no_grad():
                for q, target_q in zip(self.q_nets, self.target_q_nets):
                    for param, target_param in zip(q.parameters(), target_q.parameters()):
                        target_param.lerp_(param, self.tau)
            self._n_updates += 1
            self.last_losses = {"critic": float(critic_loss.detach()), "actor": float(actor_loss.detach())}


def build_pc_agent(cfg, env, *, training):
    return build_agent(cfg, env, training=training, agent_class=PreferenceConditionedSAC)
