import pathlib
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import torch.utils.data
from models import mri_utils


class MICCAI2020Data(torch.utils.data.Dataset):
    # This is the same as fastMRI singlecoil_knee, except we provide a custom test split
    # and also normalize images by the mean norm of the k-space over training data
    KSPACE_WIDTH = 368
    KSPACE_HEIGHT = 640
    START_PADDING = 166
    END_PADDING = 202
    CENTER_CROP_SIZE = 320
    NORMALIZING_FACTOR = 0.1520357 * 2

    def __init__(
        self,
        root: pathlib.Path,
        num_cols: Optional[int] = None,
        num_volumes: Optional[int] = None,
        num_rand_slices: Optional[int] = None,
        custom_split: Optional[str] = None,
    ):
        # self.transform = transform
        self.examples: List[Tuple[pathlib.PurePath, int]] = []

        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState(1234)

        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            data = h5py.File(fname, "r")
            if num_cols is not None and data["kspace"].shape[2] != num_cols:
                continue
            files.append(fname)

        if custom_split is not None:
            split_info = []
            with open(f"datasets/{custom_split}.txt") as f:
                for line in f:
                    split_info.append(line.rsplit("\n")[0])
            files = [f for f in files if f.name in split_info]

        if num_volumes is not None:
            self.rng.shuffle(files)
            files = files[:num_volumes]

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, "r")
            kspace = data["kspace"]

            if num_rand_slices is None:
                num_slices = kspace.shape[0]
                self.examples += [(fname, slice_id) for slice_id in range(num_slices)]
            else:
                slice_ids = list(range(kspace.shape[0]))
                self.rng.seed(seed=volume_i)
                self.rng.shuffle(slice_ids)
                self.examples += [
                    (fname, slice_id) for slice_id in slice_ids[:num_rand_slices]
                ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice_id]
            kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
            kspace = mri_utils.ifftshift(kspace, dim=(-3, -2))
            target = torch.view_as_real(
                        torch.fft.ifftn(
                            torch.view_as_complex(kspace), dim=(-2, -1), norm="backward"
                        )
                    )
            target = mri_utils.ifftshift(target, dim=(-3, -2))
            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07
            target /= MICCAI2020Data.NORMALIZING_FACTOR

            return target.permute(2, 0, 1), kspace.permute(2, 0, 1)


if __name__ == "__main__":
    dataset = MICCAI2020Data(
            "data/singlecoil_val/",
            custom_split="raw_test",
            num_cols=368)


