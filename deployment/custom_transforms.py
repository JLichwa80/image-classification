
import numpy as np
import cv2
import random
import torch
from fastai.vision.all import PILImage, ItemTransform

# Image Transformation Settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_IMAGE_BLUR = 7

COLORMAP_SELECTION = 'HOT'

class EnsureGrayscale(ItemTransform):
  """Convert image to grayscale using pure numpy, then to 3-channel for ResNet"""

  def __repr__(self):
      return f"{self.__class__.__name__}()"

  def encodes(self, x):
      is_tuple = isinstance(x, (tuple, list))
      img = x[0] if is_tuple else x
      label = x[1] if is_tuple and len(x) > 1 else None

      arr = np.array(img)

      # Convert to grayscale using numpy formula
      if len(arr.shape) == 2:
          # Already grayscale
          arr_gray = arr
      elif len(arr.shape) == 3:
          # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
          arr_gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
      else:
          arr_gray = arr

      # Replicate to 3 channels for ResNet
      arr_3ch = np.stack([arr_gray, arr_gray, arr_gray], axis=-1)
      res = PILImage.create(arr_3ch)

      if label is not None:
          return (res, label)
      else:
          return (res,)


# Apply CLAHE to grayscale image
class CLAHETransform(ItemTransform):

    def __init__(self, p=1.0):
        self.clip_limit = CLAHE_CLIP_LIMIT
        self.tile_grid_size = CLAHE_TILE_GRID_SIZE
        self.medianBlur = CLAHE_IMAGE_BLUR
        self.p = p

    def __repr__(self):
      return (f"{self.__class__.__name__}("
              f"clip_limit={self.clip_limit}, "
              f"tile_grid_size={self.tile_grid_size}, "
              f"medianBlur={self.medianBlur}, "
              f"p={self.p})")

    def encodes(self, x):
        is_tuple = isinstance(x, (tuple, list))
        img = x[0] if is_tuple else x
        label = x[1] if is_tuple and len(x) > 1 else None

        if random.random() > self.p:
            return x

        arr = np.array(img)

        # Get grayscale
        if len(arr.shape) == 2:
            gray = arr
        elif len(arr.shape) == 3:
            gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = arr

        gray = cv2.medianBlur(gray, self.medianBlur)
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        gray = clahe.apply(gray)

        # Replicate to 3 channels
        arr_3ch = np.stack([gray, gray, gray], axis=-1)
        res = PILImage.create(arr_3ch)

        if label is not None:
            return (res, label)
        else:
            return (res,)

# Apply colormap to grayscale image
class ColormapTransform(ItemTransform):
    """Apply colormap to grayscale image"""
    def __init__(self, p=1.0, colormap=COLORMAP_SELECTION):
        self.colormap = colormap
        self.p = p
        self.cv2_colormaps = {
            'JET': cv2.COLORMAP_JET,
            'HOT': cv2.COLORMAP_HOT,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS,
            'PLASMA': cv2.COLORMAP_PLASMA,
            'OCEAN': cv2.COLORMAP_OCEAN,
            'BONE': cv2.COLORMAP_BONE,
            'WINTER': cv2.COLORMAP_WINTER,
            'INFERNO': cv2.COLORMAP_INFERNO,
            'MAGMA': cv2.COLORMAP_MAGMA,
        }

    def encodes(self, x):
        is_tuple = isinstance(x, (tuple, list))
        img = x[0] if is_tuple else x
        label = x[1] if is_tuple and len(x) > 1 else None

        if random.random() > self.p:
            return x

        arr = np.array(img)

        # Get grayscale
        if len(arr.shape) == 2:
            gray = arr
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Normalize to 0-255 if needed
        if gray.dtype != np.uint8:
            gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

        # Apply colormap
        if self.colormap in self.cv2_colormaps:
            colored = cv2.applyColorMap(gray, self.cv2_colormaps[self.colormap])
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        else:
            colored = np.stack([gray, gray, gray], axis=-1)

        res = PILImage.create(colored)

        if label is not None:
            return (res, label)
        else:
            return (res,)
    def __repr__(self):
      return (f"{self.__class__.__name__}("
              f"colormap='{self.colormap}', "
              f"p={self.p})")


# Loss function with focus on most difficult images
class FastFocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha: float or 1D tensor of shape [num_classes]
        gamma: focusing parameter
        """
        super().__init__()
        # register alpha as buffer so it moves with the module to cuda
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        logp = torch.nn.functional.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()

        # alpha per sample
        if self.alpha.ndim == 0:
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]

        focal_loss = -alpha_t * (1 - p_t) ** self.gamma * logp_t

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    def activation(self, x):
        return torch.nn.functional.softmax(x, dim=1)

    def decodes(self, x):
        return x.argmax(dim=1)
