"""Explicit shared-protocol NYU dataset, not full official training reproduction.

PIL bilinear RGB/depth resizing and 228x304 center crop match the audited
reference evaluation geometry for 640x480 inputs. Training retains this
project's random crop/flip/brightness augmentation. RGB stays in [0,1]; an
external pretrained-model adapter must apply its required normalization.
"""
import h5py
import numpy as np
from PIL import Image
import torch

from train_nyuv2_hdf5 import NYUv2HDF5Dataset


class NYUPILSharedDataset(NYUv2HDF5Dataset):
    def __getitem__(self, idx):
        with h5py.File(self.file_list[idx], "r") as stream:
            rgb = stream["rgb"][:].transpose(1, 2, 0)
            depth = stream["depth"][:].astype(np.float32)
        if rgb.shape[:2] != (480, 640) or depth.shape != (480, 640):
            raise ValueError("Protocol requires raw 480x640 RGB/depth pairs")
        size = (self.resize_size[1], self.resize_size[0])
        rgb = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        depth = Image.fromarray(depth).resize(size, Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.array(rgb, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255.
        depth = torch.from_numpy(np.array(depth, dtype=np.float32, copy=True))[None]
        rgb, depth = self._crop(rgb, depth)
        generator = None if self.use_aug else torch.Generator().manual_seed(self.sparse_seed + idx)
        sparse = self._sample_sparse(depth, generator)
        if self.use_aug:
            rgb, depth, sparse = self._augment(rgb, depth, sparse)
        return {"rgb": rgb, "depth_gt": depth, "sparse_depth": sparse,
                "normal_gt": torch.zeros(3, *self.crop_size)}
