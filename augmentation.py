import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

class RandomShiftAug(nn.Module):
  """DrQ-style random shift augmentation (replicate pad + random crop)."""

  def __init__(self, pad: int = 4):
    super().__init__()
    self.pad = int(max(0, pad))

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    if self.pad == 0:
      return image_batch
    if image_batch.dim() != 4:
      raise ValueError(
          f"Expected 4D tensor (batch, channels, height, width) for random shift, got shape {tuple(image_batch.shape)}"
      )

    batch_size, num_channels, img_height, img_width = image_batch.shape

    # Pad image borders by replicating edge pixels to avoid artifacts at boundaries
    padded_image = F.pad(image_batch, (self.pad, self.pad,
                         self.pad, self.pad), mode='replicate')

    # Generate random crop offset to shift the image by a random amount between -pad and +pad
    num_possible_shifts = 2 * self.pad + 1
    col_shift = torch.randint(0, num_possible_shifts, (batch_size,),
                              device=image_batch.device, dtype=torch.long)
    row_shift = torch.randint(0, num_possible_shifts, (batch_size,),
                              device=image_batch.device, dtype=torch.long)

    padded_width = img_width + 2 * self.pad

    # Row gather must keep full padded width so column gather still has valid range to index
    row_indices = torch.arange(
        img_height, device=image_batch.device, dtype=torch.long).view(1, 1, img_height, 1)
    row_indices = row_indices.expand(
        batch_size, num_channels, img_height, padded_width) + row_shift.view(batch_size, 1, 1, 1)

    # Column gather reduces padded width back to original width using the random col offset
    col_indices = torch.arange(
        img_width, device=image_batch.device, dtype=torch.long).view(1, 1, 1, img_width)
    col_indices = col_indices.expand(
        batch_size, num_channels, img_height, img_width) + col_shift.view(batch_size, 1, 1, 1)

    # Gather rows first (output: B,C,H,padded_W), then columns (output: B,C,H,W)
    shifted_image = torch.gather(padded_image, dim=2, index=row_indices)
    shifted_image = torch.gather(shifted_image, dim=3, index=col_indices)
    return shifted_image

class DrQDictFeaturesExtractor(BaseFeaturesExtractor):
  """Multi-input extractor with optional DrQ augmentation on image_front only."""

  def __init__(
      self,
      observation_space: spaces.Dict,
      cnn_output_dim: int = 256,
      drq_enabled: bool = False,
      drq_pad: int = 4,
      normalized_image: bool = False,
  ):
    super().__init__(observation_space, features_dim=1)

    if not isinstance(observation_space, spaces.Dict):
      raise TypeError(
          "DrQDictFeaturesExtractor expects a Dict observation space")

    self.image_key = "image"
    if self.image_key not in observation_space.spaces:
      raise KeyError(
          f"Expected '{self.image_key}' in observation space keys {list(observation_space.spaces.keys())}"
      )

    self.drq_enabled = bool(drq_enabled)
    self.random_shift = RandomShiftAug(pad=drq_pad)
    self.extractors = nn.ModuleDict()

    # Build separate extractors for each observation key
    total_feature_size = 0
    for obs_key, obs_subspace in observation_space.spaces.items():
      if obs_key == self.image_key:
        # Image gets CNN encoder to produce learned features
        self.extractors[obs_key] = NatureCNN(
            obs_subspace,
            features_dim=cnn_output_dim,
            normalized_image=normalized_image,
        )
        total_feature_size += cnn_output_dim
      else:
        # Inputs like goal, traffic_light, distance_to_stop are flattened
        self.extractors[obs_key] = nn.Flatten()
        total_feature_size += int(np.prod(obs_subspace.shape))

    self._features_dim = total_feature_size

  def set_drq_enabled(self, enabled: bool) -> None:
    self.drq_enabled = bool(enabled)

  @staticmethod
  def _to_channel_first(image: torch.Tensor) -> torch.Tensor:
    """Convert NHWC input to NCHW when required."""
    if image.dim() == 3:
      image = image.unsqueeze(0)
    if image.dim() != 4:
      raise ValueError(
          f"Expected 4D image tensor, got shape {tuple(image.shape)}")

    # Check if the image is already in channel-first format
    if image.shape[1] in (1, 3, 4):
      return image
    # Check if the image is in channel-last format and permute to channel-first
    if image.shape[-1] in (1, 3, 4):
      return image.permute(0, 3, 1, 2)
    raise ValueError(
        f"Cannot infer channel dimension for image shape {tuple(image.shape)}")

  def forward(self, observations) -> torch.Tensor:
    # Process each observation component through its extractor
    encoded_features = []
    for obs_key, extractor in self.extractors.items():
      obs_tensor = observations[obs_key]
      if obs_key == self.image_key:
        obs_tensor = self._to_channel_first(obs_tensor)
        # Apply random shift augmentation only during training if DrQ is enabled
        if self.training and self.drq_enabled:
          obs_tensor = self.random_shift(obs_tensor)
      encoded_features.append(extractor(obs_tensor))

    # Concatenate all encoded features into single vector for SAC
    return torch.cat(encoded_features, dim=1)
