import copy
import h5py
import math
import numpy as np
import os
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import readpcd
from utils import random_select_points, shift_point_cloud, jitter_point_cloud, \
    generate_random_rotation_matrix, generate_random_tranlation_vector, \
    transform, random_crop, shuffle_pc, random_scale_point_cloud, flip_pc, inv_R_t


def pc_normalize(pc):
    
    return pc


class CustomData(Dataset):
    def __init__(self, root, split, npts, p_keep, noise, unseen, ao=False, normal=False):
        super(CustomData, self).__init__()
        self.single = False
        assert split in ['train', 'val', 'test']
        self.split = split
        self.npts = npts
        self.p_keep = p_keep
        self.noise = noise
        self.unseen = unseen
        self.ao = ao
        self.normal = normal

        # Read all files from the 'ref' folder as target point clouds
        self.tgt_files = [os.path.join(root, 'ref', item) for item in sorted(os.listdir(os.path.join(root, 'ref')))]

        # Read all files from the corresponding split folder as source point clouds
        self.src_files = [os.path.join(root, split, item) for item in sorted(os.listdir(os.path.join(root, split)))]
        self.npts = npts
        self.p_keep = p_keep

    def __len__(self):
        return len(self.src_files)

    def compose(self, item, src_cloud, tgt_cloud):
        actual_R_t = np.array([[ 0.968255579472, 0.204919964075, -0.143139615655, 0.367150276899],
                               [-0.175158590078, 0.964767396450, 0.196324601769 ,-0.235054686666],
                               [0.178327277303, -0.165020257235, 0.970034897327 ,0.111320428550],
                               [0.000000000000, 0.000000000000, 0.000000000000 ,1.000000000000]])
        if self.split != 'train':
            inv_Rt = actual_R_t.copy()
            inv_R = inv_Rt[:3, :3]
            inv_t = inv_Rt[:3, 3]
            R = np.linalg.inv(inv_R.astype(np.float32))
            t = -np.dot(R, inv_t.astype(np.float32))
            src_cloud = random_select_points(src_cloud, m=self.npts)

        else:
            tgt_cloud = flip_pc(tgt_cloud)
            R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
            src_cloud_points = transform(src_cloud[:, :3], R, t)
            src_cloud_normal = transform(src_cloud[:, 3:], R)
            src_cloud = np.concatenate([src_cloud_points, src_cloud_normal], axis=-1)
            src_cloud = random_select_points(src_cloud, m=self.npts)

        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        if self.split == 'train' or self.noise:
            src_cloud[:, :3] = jitter_point_cloud(src_cloud[:, :3])
            tgt_cloud[:, :3] = jitter_point_cloud(tgt_cloud[:, :3])
        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_file = self.src_files[item]
        tgt_file = self.tgt_files[item % len(self.tgt_files)]

        src_points = readpcd(src_file, rtype='npy')
        tgt_points = readpcd(tgt_file, rtype='npy')

        src_cloud = np.concatenate([src_points, pc_normalize(src_points)], axis=-1)
        tgt_cloud = np.concatenate([tgt_points, pc_normalize(tgt_points)], axis=-1)

        src_cloud, tgt_cloud, R, t = self.compose(item=item, src_cloud=src_cloud, tgt_cloud=tgt_cloud)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

