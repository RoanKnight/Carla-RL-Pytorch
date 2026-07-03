from __future__ import annotations

from typing import Tuple

import torch as th
from torch import nn

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, TensorDict

_DRQ_IMAGE_KEY = "image"

class DrQDictReplayBuffer(DictReplayBuffer):
  """Dict replay buffer with helpers for DrQ two-view augmentation."""

  def __init__(self, *args, image_key: str = "image", **kwargs):
    # DrQ expects a single image observation branch with a fixed key.
    if str(image_key) != _DRQ_IMAGE_KEY:
      raise ValueError(
        f"DrQDictReplayBuffer only supports image key '{_DRQ_IMAGE_KEY}', got '{image_key}'.")
    self.image_key = _DRQ_IMAGE_KEY
    super().__init__(*args, **kwargs)

    # Fail early if the replay buffer does not actually contain image observations.
    if self.image_key not in self.observations:
      raise KeyError(
        f"DrQ image key '{self.image_key}' missing from replay observations: {list(self.observations.keys())}"
      )

  @staticmethod
  def _to_channel_first(image_batch: th.Tensor) -> Tuple[th.Tensor, bool]:
    """Return image batch as NCHW and whether original layout was NHWC."""
    # DrQ augmentations expect a 4D batch of images.
    if image_batch.dim() != 4:
      raise ValueError(
        f"Expected 4D image tensor for DrQ augmentation, got shape {tuple(image_batch.shape)}")

    # If the channel dimension is already in position 1, leave the tensor as-is.
    if image_batch.shape[1] in (1, 3, 4):
      return image_batch, False
    # Otherwise treat the input as NHWC and convert it to NCHW.
    if image_batch.shape[-1] in (1, 3, 4):
      return image_batch.permute(0, 3, 1, 2), True

    raise ValueError(
      f"Could not infer channel dimension for image tensor shape {tuple(image_batch.shape)}")

  @staticmethod
  def _augment_image(image_batch: th.Tensor, augmenter: nn.Module) -> th.Tensor:
    # Normalize layout, apply the augmentation, then restore the original format if needed.
    image_nchw, was_channel_last = DrQDictReplayBuffer._to_channel_first(image_batch)
    augmented = augmenter(image_nchw)
    if was_channel_last:
      return augmented.permute(0, 2, 3, 1)
    return augmented

  @staticmethod
  def _copy_tensor_dict(tensor_dict: TensorDict) -> TensorDict:
    # Shallow-copy the dict so the sampled batch can be rewritten safely.
    return {key: value for key, value in tensor_dict.items()}

  def create_augmented_views(
    self,
    replay_data: DictReplayBufferSamples,
    augmenter: nn.Module,
  ) -> tuple[DictReplayBufferSamples, DictReplayBufferSamples]:
    """Build two independent DrQ views from one sampled transition batch."""
    # Both the current and next observations must include the image branch.
    if self.image_key not in replay_data.observations:
      raise KeyError(f"Missing image key '{self.image_key}' in sampled observations.")
    if self.image_key not in replay_data.next_observations:
      raise KeyError(f"Missing image key '{self.image_key}' in sampled next observations.")

    # Only augment the image key; the rest of the observation dict is shared across views and should not be modified.
    obs_image = replay_data.observations[self.image_key]
    next_obs_image = replay_data.next_observations[self.image_key]

    view1_obs = self._copy_tensor_dict(replay_data.observations)
    view1_next_obs = self._copy_tensor_dict(replay_data.next_observations)
    view1_obs[self.image_key] = self._augment_image(obs_image, augmenter)
    view1_next_obs[self.image_key] = self._augment_image(next_obs_image, augmenter)

    view2_obs = self._copy_tensor_dict(replay_data.observations)
    view2_next_obs = self._copy_tensor_dict(replay_data.next_observations)
    view2_obs[self.image_key] = self._augment_image(obs_image, augmenter)
    view2_next_obs[self.image_key] = self._augment_image(next_obs_image, augmenter)

    # Return two image-augmented views of the same replay batch for DrQ critic updates.
    return (
      replay_data._replace(
        observations=view1_obs,
        next_observations=view1_next_obs,
      ),
      replay_data._replace(
        observations=view2_obs,
        next_observations=view2_next_obs,
      ),
    )
