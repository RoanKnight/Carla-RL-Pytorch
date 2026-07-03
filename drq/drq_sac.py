from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

from augmentation import RandomShiftAug
from drq.drq_replay_buffer import DrQDictReplayBuffer

_DRQ_IMAGE_KEY = "image"

class DrQSAC(SAC):
  """SAC variant with DrQ-v1 two-view random-shift critic updates."""

  def __init__(
    self,
    *args,
    drq_pad: int = 4,
    drq_num_views: int = 2,
    drq_image_key: str = "image",
    **kwargs,
  ):
    # DrQ-v1 always uses exactly two augmented views of the same replay sample.
    if int(drq_num_views) != 2:
      raise ValueError(f"DrQ-v1 expects exactly 2 augmented views, got {drq_num_views}.")
    # Ensure the image key exists and is consistent with the replay buffer's expected key.
    if str(drq_image_key) != _DRQ_IMAGE_KEY:
      raise ValueError(f"DrQSAC only supports image key '{_DRQ_IMAGE_KEY}', got '{drq_image_key}'.")

    self.drq_pad = int(max(0, drq_pad))
    self.drq_num_views = int(drq_num_views)
    self.drq_image_key = _DRQ_IMAGE_KEY
    self.random_shift: RandomShiftAug | None = None

    super().__init__(*args, **kwargs)

  def _setup_model(self) -> None:
    super()._setup_model()

    # DrQ requires a Dict observation space with a specific image key for augmentation.
    if not isinstance(self.observation_space, spaces.Dict):
      raise TypeError("DrQSAC requires a Dict observation space.")
    if self.drq_image_key not in self.observation_space.spaces:
      raise KeyError(f"DrQSAC expected image key '{self.drq_image_key}' in observation space keys "
                     f"{list(self.observation_space.spaces.keys())}")

    # Use the same random shift augmentation for both critic updates and replay buffer view sampling.
    self.random_shift = RandomShiftAug(self.drq_pad).to(self.device)

  def _sample_drq_views(
    self, replay_data: DictReplayBufferSamples
  ) -> tuple[DictReplayBufferSamples, DictReplayBufferSamples]:
    # Create two independently shifted views from the same sampled batch.
    if self.random_shift is None:
      raise RuntimeError("Random shift augmentation was not initialized.")
    if not isinstance(self.replay_buffer, DrQDictReplayBuffer):
      raise TypeError("DrQSAC requires DrQDictReplayBuffer for DrQ two-view sampling.")
    return self.replay_buffer.create_augmented_views(replay_data, self.random_shift)

  def _compute_target_q(
    self,
    replay_data: DictReplayBufferSamples,
    ent_coef: th.Tensor,
    discounts: float | th.Tensor,
  ) -> th.Tensor:
    # Use the target critic on the next observation to form the SAC bootstrapped target.
    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    return replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

  def train(self, gradient_steps: int, batch_size: int = 64) -> None:
    self.policy.set_training_mode(True)

    # Update all optimizers together so the learning rate schedule stays in sync.
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      optimizers.append(self.ent_coef_optimizer)
    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
      # Sample a replay batch, then build two DrQ views from it.
      replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
      if not isinstance(replay_data.observations, dict):
        raise TypeError("DrQSAC only supports Dict replay samples.")

      replay_view_a, replay_view_b = self._sample_drq_views(replay_data)
      discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

      if self.use_sde:
        self.actor.reset_noise()

      # Actor update uses one augmented view while the critic sees both views.
      actions_pi, log_prob = self.actor.action_log_prob(replay_view_a.observations)
      log_prob = log_prob.reshape(-1, 1)

      ent_coef_loss = None
      if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
        ent_coef = th.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())
      else:
        ent_coef = self.ent_coef_tensor

      ent_coefs.append(ent_coef.item())

      if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

      with th.no_grad():
        # Average the two augmented target estimates to get a more stable target for the critic update.
        target_q_a = self._compute_target_q(replay_view_a, ent_coef, discounts)
        target_q_b = self._compute_target_q(replay_view_b, ent_coef, discounts)
        target_q_values = 0.5 * (target_q_a + target_q_b)

      # Fit the critic on both augmented observations against the shared target.
      current_q_values_a = self.critic(replay_view_a.observations, replay_data.actions)
      current_q_values_b = self.critic(replay_view_b.observations, replay_data.actions)

      critic_loss_a = 0.5 * sum(
        F.mse_loss(current_q, target_q_values) for current_q in current_q_values_a)
      critic_loss_b = 0.5 * sum(
        F.mse_loss(current_q, target_q_values) for current_q in current_q_values_b)
      critic_loss = critic_loss_a + critic_loss_b
      critic_losses.append(critic_loss.item())

      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()

      # Improve the policy by maximizing Q minus entropy cost.
      q_values_pi = th.cat(self.critic(replay_view_a.observations, actions_pi), dim=1)
      min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
      actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
      actor_losses.append(actor_loss.item())

      self.actor.optimizer.zero_grad()
      actor_loss.backward()
      self.actor.optimizer.step()

      # Update the target critic and batch norm stats with a Polyak averaging factor.
      if gradient_step % self.target_update_interval == 0:
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/ent_coef", np.mean(ent_coefs))
    self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    self.logger.record("train/drq_pad", float(self.drq_pad))
    if len(ent_coef_losses) > 0:
      self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
