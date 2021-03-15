import random
import math
import numpy as np
import time



def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def flip(joints, axis=0):
    num_v = joints.shape[2]
    cx, cy, _ = np.mean(joints, axis=2)
    cx = np.expand_dims(cx, axis=2).repeat(num_v, axis=1)
    cy = np.expand_dims(cy, axis=2).repeat(num_v, axis=1)
    joints_ct = joints.copy()
    joints_ct[0, :, :, :] = joints_ct[0, :, :, :] - cx
    joints_ct[1, :, :, :] = joints_ct[1, :, :, :] - cy
    # flip
    joints_ct[axis, :, :, :] = 0.0 - joints_ct[axis, :, :, :]
    joints[0, :, :, :] = joints_ct[0, :, :, :] + cx
    joints[1, :, :, :] = joints_ct[1, :, :, :] + cy
    return joints

def augmentation(joints, axis=0, scale=[0.8, 1.2], degree=20):
    def rotate(origin, point, angle):
        ox, oy = origin
        px, py = point[0, :, :, :], point[1, :, :, :]
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    num_v = joints.shape[2]
    rescale = np.random.uniform(scale[0], scale[1])
    cx, cy, _ = np.mean(joints, axis=2)
    cx = np.expand_dims(cx, axis=2).repeat(num_v, axis=1)
    cy = np.expand_dims(cy, axis=2).repeat(num_v, axis=1)
    joints_ct = joints.copy()
    joints_ct[0, :, :, :] = joints_ct[0, :, :, :] - cx
    joints_ct[1, :, :, :] = joints_ct[1, :, :, :] - cy
    # rescale
    joints_ct = joints_ct * rescale
    # flip
    if np.random.randint(2) == 0:
        joints_ct[axis, :, :, :] = 0.0 - joints_ct[axis, :, :, :]
    # rotation
    angle = np.random.randint(0-degree, degree)
    angle = math.pi * 2 * (angle / 360)
    rot_x, rot_y = rotate((0, 0), joints_ct, angle)
    joints[0, :, :, :] = rot_x + cx
    joints[1, :, :, :] = rot_y + cy
    # joints[0, :, :, :] = joints_ct[0, :, :, :] + cx
    # joints[1, :, :, :] = joints_ct[1, :, :, :] + cy
    return joints


def rescale(joints, scale=0.8, shift=[0.0, 0.0]):
    num_v = joints.shape[2]
    cx, cy, _ = np.mean(joints, axis=2)
    cx = np.expand_dims(cx, axis=2).repeat(num_v, axis=1)
    cy = np.expand_dims(cy, axis=2).repeat(num_v, axis=1)
    cx = cx + shift[0]
    cy = cy + shift[1]
    joints_ct = joints.copy()
    joints_ct[0, :, :, :] = joints_ct[0, :, :, :] - cx
    joints_ct[1, :, :, :] = joints_ct[1, :, :, :] - cy
    joints_ct = joints_ct * scale
    joints[0, :, :, :] = joints_ct[0, :, :, :] + cx
    joints[1, :, :, :] = joints_ct[1, :, :, :] + cy
    return joints


def rotation(joints, degree=15):
    np.random.seed(int(time.time()))
    # degree = math.pi * 2 * (degree / 360)
    # angle = math.pi * 2 * (degree / 360)
    def rotate(origin, point, angle):
        ox, oy = origin
        px, py = point[0, :, :], point[1, :, :]
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy
    num_v = joints.shape[2]
    cx, cy, _ = np.mean(joints, axis=2)
    cx = np.expand_dims(cx, axis=2).repeat(num_v, axis=1)
    cy = np.expand_dims(cy, axis=2).repeat(num_v, axis=1)
    joints_ct = joints.copy()
    joints_ct[0, :, :, :] = joints_ct[0, :, :, :] - cx
    joints_ct[1, :, :, :] = joints_ct[1, :, :, :] - cy
    angle = np.random.randint(0-degree, degree)
    angle = math.pi * 2 * (angle / 360)
    for i in range(joints.shape[1]):
        # angle = np.random.randint(0-degree, degree)
        # angle = math.pi * 2 * (angle / 360)
        rot_x, rot_y = rotate((0, 0), joints_ct[:, i, :, :], angle)
        joints[0, i, :, :] = rot_x + cx[i, :, :]
        joints[1, i, :, :] = rot_y + cy[i, :, :]
    return joints.copy()


# def rotation(joints, degree=15):
#     degree = math.pi * 2 * (degree / 360)
#     def rotate(origin, point, angle):
#         ox, oy = origin[0, :, :, :], origin[1, :, :, :]
#         px, py = point[0, :, :, :], point[1, :, :, :]
#     
#         qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
#         qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
#         return qx, qy
#     num_v = joints.shape[2]
#     joints_ct = (joints[:2, :, 8, :] + joints[:2, :, 9, :]) / 2
#     joints_ct = np.expand_dims(joints_ct, axis=2).repeat(num_v, axis=2)
#     rot_x, rot_y = rotate(joints_ct, joints, degree)
#     joints[0, :, :, :] = rot_x
#     joints[1, :, :, :] = rot_y
#     return joints.copy()


def temporal_aug(joints, proportion=0.1):
    nframe = joints.shape[1]
    point_sum = np.sum(joints, axis=(0, 2, 3))
    valid_num = np.sum(point_sum != 0)
    frame_id = np.arange(valid_num).tolist()
    supp_num = nframe - valid_num
    random.shuffle(frame_id)
    frame_id += frame_id[:valid_num]
    frame_id = np.sort(frame_id)
    joints_supp = joints[:, frame_id, :, :].copy()
    return joints_supp


def temporal_pad(joints, proportion=0.1):
    nframe = joints.shape[1]
    point_sum = np.sum(joints, axis=(0, 2, 3))
    valid_num = np.sum(point_sum != 0)
    frame_id = np.arange(valid_num).tolist()
    supp_num = nframe % valid_num
    repeat_num = nframe // valid_num
    mid_l = (valid_num - supp_num) // 2
    mid_r = mid_l + supp_num
    frame_supp = frame_id[mid_l:mid_r]
    frame_id_last = []
    for i in range(repeat_num):
        frame_id_last += frame_id
    frame_id_last += frame_supp
    frame_id_last = np.sort(frame_id_last)
    joints_pad = joints[:, frame_id_last, :, :].copy()
    return joints_pad


