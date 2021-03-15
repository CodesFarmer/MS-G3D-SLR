import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

num_node = 73
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (0, 6),
          (5, 7), (7, 8), (5, 9), (6, 10), (10, 11), (6, 12),
          (8, 13), (13, 14), (14, 15), (15, 16),
          (8, 17), (17, 18), (18, 19), (19, 20),
          (8, 21), (21, 22), (22, 23), (23, 24),
          (8, 25), (25, 26), (26, 27), (27, 28),
          (8, 29), (29, 30), (30, 31), (31, 32),
          (11, 33), (33, 34), (34, 35), (35, 36),
          (11, 37), (37, 38), (38, 39), (39, 40),
          (11, 41), (41, 42), (42, 43), (43, 44),
          (11, 45), (45, 46), (46, 47), (47, 48),
          (11, 49), (49, 50), (50, 51), (51, 52), (0, 53),
          (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59),
          (53, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65),
          (53, 66), (53, 67), (6, 68), (5, 69),
          (54, 70), (10, 71), (7, 72)
          ]
# num_node = 58
# self_link = [(i, i) for i in range(num_node)]
# inward = [(0, 1), (15, 0), (17, 15), (14, 0), (16, 14), (5, 1), (2, 1), # part neck 0-6
#       (6, 5), (7, 6), (11, 5), (3, 2), (4, 3), (8, 2), # body and limb 7-12
#       (18, 7), (19, 18), (20, 19), (21, 20), # 13-16
#       (22, 7), (23, 22), (24, 23), (25, 24), # 17-20
#       (26, 7), (27, 26), (28, 27), (29, 28), # 21-24
#       (30, 7), (31, 30), (32, 31), (32, 33), # 25-28
#       (34, 7), (35, 34), (36, 35), (37, 36), # 29-32
#       (38, 4), (39, 38), (40, 39), (41, 40), # 33-36
#       (42, 4), (43, 42), (44, 43), (45, 44), # 37-40
#       (46, 4), (47, 46), (48, 47), (49, 48), # 41-44
#       (50, 4), (51, 50), (52, 51), (53, 52), # 45-48
#       (54, 4), (55, 54), (56, 55), (57, 55), (0, 0), # 49-53
#       (7, 0), (21, 7), (25, 7), (29, 7), (32, 7), (37, 7), # 54-59
#       (4, 0), (41, 4), (45, 4), (49, 4), (53, 4), (57, 4), # 60-65
#       (6, 0), (3, 0), (7, 2), (4, 5), (7, 4), (7, 3), (4, 6)] # 66-72
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
