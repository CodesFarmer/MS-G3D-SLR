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

num_node = 126
self_link = [(i, i) for i in range(num_node)]
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14), (7, 18), (18, 19), (19, 20), (20, 21), (7, 22), (22, 23), (23, 24), 
          (24, 25), (7, 26), (26, 27), (27, 28), (28, 29), (7, 30), (30, 31), (31, 32),
          (32, 33), (7, 34), (34, 35), (35, 36), (36, 37), (4, 38), (38, 39), (39, 40),
          (40, 41), (4, 42), (42, 43), (43, 44), (44, 45), (4, 46), (46, 47), (47, 48),
          (48, 49), (4, 50), (50, 51), (51, 52), (52, 53), (4, 54), (54, 55), (55, 56), (56, 57),
          (0, 58)]
face_part = []
for i in range(58, 74):
    face_part.append((i, i+1))
face_part += [(75, 76), (76, 77), (77, 78), (78, 79)]
face_part += [(80, 81), (81, 82), (82, 83), (83, 84)]
face_part += [(85, 86), (86, 87), (87, 88)]
face_part += [(89, 90), (90, 91), (91, 92), (92, 93)]
face_part += [(94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 94)]
face_part += [(100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 100)]
for i in range(106, 117):
    face_part.append((i, i+1))
face_part.append((117, 106))
for i in range(118, 125):
    face_part.append((i, i+1))
face_part.append((125, 118))
inward = inward + face_part
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
