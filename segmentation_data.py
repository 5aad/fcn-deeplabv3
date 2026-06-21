from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MASK_EXTENSIONS = {".png"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_supported_files(
    directory: Path,
    extensions: set[str],
) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def index_files_by_stem(
    paths: list[Path],
    file_kind: str,
) -> dict[str, Path]:
    index: dict[str, Path] = {}

    for path in paths:
        if path.stem in index:
            raise ValueError(
                f"Duplicate {file_kind} stem '{path.stem}': "
                f"{index[path.stem].name} and {path.name}"
            )
        index[path.stem] = path

    return index


def load_image_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask_index(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    if mask.ndim == 3:
        raise ValueError(
            f"Expected class-index mask, but found RGB/color mask: {mask_path}. "
            "A color-to-class conversion mapping is required before training."
        )

    return mask


def create_training_augmentation() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=12,
                val_shift_limit=8,
                p=0.2,
            ),
        ]
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int,
        num_classes: int,
        ignore_index: int | None = None,
        augmentation: A.Compose | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augmentation = augmentation
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        image_paths = list_supported_files(self.image_dir, IMAGE_EXTENSIONS)
        mask_paths = list_supported_files(self.mask_dir, MASK_EXTENSIONS)

        if not image_paths:
            raise FileNotFoundError(
                f"No supported images found in {self.image_dir}"
            )

        mask_index = index_files_by_stem(mask_paths, "mask")
        self.samples: list[tuple[Path, Path]] = []

        for image_path in image_paths:
            mask_path = mask_index.get(image_path.stem)

            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing mask for image: {image_path.name}"
                )

            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def _validate_mask(self, mask: np.ndarray, mask_path: Path) -> None:
        valid_values = set(range(self.num_classes))

        if self.ignore_index is not None:
            valid_values.add(self.ignore_index)

        mask_values = set(np.unique(mask).astype(int).tolist())
        invalid_values = mask_values - valid_values

        if invalid_values:
            raise ValueError(
                f"Invalid mask values in {mask_path.name}: "
                f"{sorted(invalid_values)}"
            )

    def _load_and_resize(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = load_image_rgb(image_path)
        mask = load_mask_index(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask dimensions do not match for "
                f"{image_path.name}: image={image.shape[:2]}, "
                f"mask={mask.shape[:2]}"
            )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        self._validate_mask(mask, mask_path)

        if max(self.num_classes - 1, self.ignore_index or 0) <= 255:
            mask = mask.astype(np.uint8, copy=False)

        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        if self.cache_in_memory and index in self._cache:
            cached_image, cached_mask = self._cache[index]
            image = cached_image.copy()
            mask = cached_mask.copy()
        else:
            image, mask = self._load_and_resize(image_path, mask_path)

            if self.cache_in_memory:
                self._cache[index] = (image.copy(), mask.copy())

        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            self._validate_mask(mask, mask_path)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64, copy=False)).long()

        return image_tensor, mask_tensor
