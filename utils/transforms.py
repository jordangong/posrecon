from typing import Optional, Dict, Any

import kornia.augmentation as K
import torch
from kornia.constants import BorderType
from kornia.filters import filter2d
from torch import Tensor, nn
from torchvision import transforms


class SimCLRPretrainPreTransform:
    def __init__(self, img_size: int = 224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        x_1 = self.transform(x)
        x_2 = self.transform(x)

        return x_1, x_2


def imagenet_normalization():
    return K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def cifar10_normalization():
    return K.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )


def cifar100_normalization():
    return K.Normalize(
        mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
        std=[x / 255.0 for x in [68.2, 65.4, 70.4]],
    )


def flower102_normalization():
    return K.Normalize(
        mean=[x / 255.0 for x in [110.4, 97.4, 75.6]],
        std=[x / 255.0 for x in [75.1, 62.9, 69.7]],
    )


def oxford_iiit_pet_normalization():
    return K.Normalize(
        mean=[x / 255.0 for x in [122., 113.7, 100.9]],
        std=[x / 255.0 for x in [68.3, 67.1, 68.8]],
    )


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
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))
        self.flags = dict(
            kernel_size=kernel_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            border_type=BorderType.get(border_type),
        )

    @staticmethod
    def batch_gaussian(window_size: int, sigma: Tensor) -> Tensor:
        device, dtype = sigma.device, sigma.dtype
        x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        if window_size % 2 == 0:
            x = x + 0.5
        gauss_pow = -x ** 2.0 / (2 * sigma ** 2).unsqueeze(-1)
        return torch.softmax(gauss_pow, dim=-1)

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
        out_x = filter2d(input, kernel_x.unsqueeze(1), border_type)
        out = filter2d(out_x, kernel_y.unsqueeze(-1), border_type)
        return out

    def apply_transform(
            self,
            input: Tensor,
            params: Dict[str, Tensor],
            flags: Dict[str, Any],
            transform: Optional[Tensor] = None
    ) -> Tensor:
        sigma_dist = torch.distributions.Uniform(self.sigma_min, self.sigma_max)
        sigma = sigma_dist.sample((input.size(0),))
        return self.batch_gaussian_blur2d(
            input,
            kernel_size=(flags["kernel_size"], flags["kernel_size"]),
            sigma=(sigma, sigma),
            border_type=flags["border_type"].name.lower(),
        )


class SimCLRPretrainPostTransform(nn.Module):
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
            self.transform = K.AugmentationSequential(*data_transforms)
        else:
            self.transform = K.AugmentationSequential(*data_transforms, normalize)

    @torch.no_grad()
    def forward(self, x) -> Tensor:
        return self.transform(x)


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
