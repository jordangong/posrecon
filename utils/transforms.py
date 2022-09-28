from typing import Optional, Dict, Any

import kornia.augmentation as K
import torch
from kornia.constants import BorderType
from kornia.filters import filter2d_separable
from torch import Tensor, nn


def imagenet_normalization():
    return K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class RandomSigmaGaussianBlur(K.IntensityAugmentationBase2D):
    """
    Implements Gaussian blur as described in the SimCLR paper
    Adapted from torchvision and opencv to kornia
    """

    def __init__(
            self,
            kernel_size: int,
            sigma_min: float = 0.1,
            sigma_max: float = 2.0,
            border_type: str = "reflect",
            same_on_batch: bool = False,
            p: float = 0.5,
            keepdim: bool = False,
            return_transform: Optional[bool] = None,
    ):
        super().__init__(
            p=p,
            return_transform=return_transform,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )
        self.flags = dict(
            kernel_size=kernel_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            border_type=BorderType.get(border_type),
        )
        self.sigma_dist = torch.distributions.Uniform(sigma_min, sigma_max)

    @staticmethod
    def batch_gaussian(window_size: int, sigma: Tensor) -> Tensor:
        device, dtype = sigma.device, sigma.dtype
        x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        if window_size % 2 == 0:
            x = x + 0.5
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2).unsqueeze(-1))
        return gauss / gauss.sum(-1, keepdim=True)

    def get_batch_gaussian_kernel1d(self, kernel_size: int, sigma: Tensor) -> torch.Tensor:
        if not isinstance(kernel_size, int) \
                or (kernel_size % 2 == 0) \
                or (kernel_size <= 0):
            raise TypeError("kernel_size must be an odd positive integer. "
                            "Got {}".format(kernel_size))
        window_1d: torch.Tensor = self.batch_gaussian(kernel_size, sigma)
        return window_1d

    def batch_gaussian_blur2d(
            self,
            input: Tensor,
            kernel_size: tuple[int, int],
            sigma: tuple[Tensor, Tensor],
            border_type: str = 'reflect',
    ) -> torch.Tensor:
        kernel_x: torch.Tensor = self.get_batch_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y: torch.Tensor = self.get_batch_gaussian_kernel1d(kernel_size[0], sigma[0])
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
        return out

    def apply_transform(
            self,
            input: Tensor,
            params: Dict[str, Tensor],
            flags: Dict[str, Any],
            transform: Optional[Tensor] = None
    ) -> Tensor:
        sigma = self.sigma_dist.sample((input.size(0),))
        return self.batch_gaussian_blur2d(
            input,
            kernel_size=(flags["kernel_size"], flags["kernel_size"]),
            sigma=(sigma, sigma),
            border_type=flags["border_type"].name.lower(),
        )


class SimCLRPretrainTransform(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.0,
            normalize=None,
    ) -> None:
        super().__init__()

        self.jitter_strength = jitter_strength
        self.img_size = img_size
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        data_transforms = [
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.8 * self.jitter_strength,
                contrast=0.8 * self.jitter_strength,
                saturation=0.8 * self.jitter_strength,
                hue=0.2 * self.jitter_strength,
                p=0.8,
            ),
            K.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.img_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(
                RandomSigmaGaussianBlur(kernel_size, p=0.5)
            )

        if normalize is None:
            self.train_transform = K.AugmentationSequential(*data_transforms)
        else:
            self.train_transform = K.AugmentationSequential(*data_transforms, normalize)

    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        xi = self.train_transform(x)
        xj = self.train_transform(x)

        return xi, xj


class SimCLRFinetuneTransform(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            jitter_strength: float = 1.0,
            normalize=None,
            eval_transform: bool = False,
    ) -> None:
        super().__init__()

        self.jitter_strength = jitter_strength
        self.img_size = img_size
        self.normalize = normalize

        data_transforms = []
        if not eval_transform:
            data_transforms += [
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(
                    brightness=0.8 * self.jitter_strength,
                    contrast=0.8 * self.jitter_strength,
                    saturation=0.8 * self.jitter_strength,
                    hue=0.2 * self.jitter_strength,
                    p=0.8,
                ),
                K.RandomGrayscale(p=0.2),
            ]

        if normalize is None:
            self.transform = K.AugmentationSequential(*data_transforms)
        else:
            self.transform = K.AugmentationSequential(*data_transforms, normalize)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)
