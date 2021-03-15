import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

bone_pairs = {
    'ntu/xview': ntu_skeleton_bone_pairs,
    'ntu/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_bone_pairs,
    'ntu120/xset': ntu_skeleton_bone_pairs,

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    ),
    'hand': ((0, 1), (15, 0), (17, 15), (14, 0), (16, 14), (5, 1), (2, 1), # part neck 0-6
          (6, 5), (7, 6), (11, 5), (3, 2), (4, 3), (8, 2), # body and limb 7-12
          (18, 7), (19, 18), (20, 19), (21, 20), # 13-16
          (22, 7), (23, 22), (24, 23), (25, 24), # 17-20
          (26, 7), (27, 26), (28, 27), (29, 28), # 21-24
          (30, 7), (31, 30), (32, 31), (32, 33), # 25-28
          (34, 7), (35, 34), (36, 35), (37, 36), # 29-32
          (38, 4), (39, 38), (40, 39), (41, 40), # 33-36
          (42, 4), (43, 42), (44, 43), (45, 44), # 37-40
          (46, 4), (47, 46), (48, 47), (49, 48), # 41-44
          (50, 4), (51, 50), (52, 51), (53, 52), # 45-48
          (54, 4), (55, 54), (56, 55), (57, 55), (0, 0), # 49-53
          (7, 0), (21, 7), (25, 7), (29, 7), (32, 7), (37, 7), # 54-59
          (4, 0), (41, 4), (45, 4), (49, 4), (53, 4), (57, 4), # 60-65
          (6, 0), (3, 0), (7, 2), (4, 5), (7, 4), (7, 3), (4, 6) # 66-72
    )
}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'kinetics': ('kinetics',),
    'hand':('hand',)
}

# parts = { 'train', 'val' }
parts = {'minitrain', 'minival', 'val'}
# parts = {'trainval'}
# parts = {'val'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics', 'hand'], required=True)
    args = parser.parse_args()

    # import pdb
    # pdb.set_trace()
    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)
            try:
                # data = np.load('../data/{}/testdata/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                N, C, T, V, M = data.shape
                N_edge = len(bone_pairs[benchmark])
                fp_sp = open_memmap(
                    '../data/{}/{}_data_bone.npy'.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, C, T, N_edge, M))

                # fp_sp[:, :C, :, :, :] = data
                # for v1, v2 in tqdm(bone_pairs[benchmark]):
                for bid, v_node in enumerate(bone_pairs[benchmark]):
                    v1, v2 = v_node
                    if benchmark != 'kinetics':
                        v1 -= 1
                        v2 -= 1
                    # if bid < 7:
                    #     norm_base = data[:, :, :, 0, :] - data[:, :, :, 1, :] + 1e-5
                    # elif 7 <= bid < 13:
                    #     norm_base = data[:, :, :, 5, :] - data[:, :, :, 2, :] + 1e-5
                    # elif 13 <= bid < 54:
                    #     norm_base = data[:, :, :, 0, :] - data[:, :, :, 1, :] + 1e-5
                    # fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :] / norm_base
                    # fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
                    fp_sp[:, :, :, bid, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            except Exception as e:
                print(f'Run into error: {e}')
                print(f'Skipping ({benchmark} {part})')
