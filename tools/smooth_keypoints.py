import numpy as np
import pickle
import argparse
from tqdm import tqdm
import threading
import time


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def get_parser():
    parser = argparse.ArgumentParser(description='MS-G3D points smooth')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='data file')
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.5,
        help='gaussian kernel argument')
    parser.add_argument(
        '--ksize',
        type=int,
        default=5,
        help='kernel size for smooth')
    parser.add_argument(
        '--method',
        type=str,
        default="softmax",
        help='normalize for weights')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='the path to save the smoothed data')
    parser.add_argument(
        '--num_thread',
        type=int,
        default=8,
        help='the number of threads')
    return parser


class SmoothThread(threading.Thread):
    def __init__(self, keys, sigma, method, ksize, data_in, data_out):
        super(SmoothThread, self).__init__()
        self._keys = keys
        self._sigma = sigma
        self._method = method
        self._ksize = ksize
        self._data_in = data_in
        self._data_out = data_out
        self._kernel = self.kernel()

    def run(self):
        process = tqdm(self._keys, dynamic_ncols=True)
        for skey in process:
            self._data_out[skey] = self.smooth(self._data_in[skey])

    def kernel(self):
        def norm(vector):
            return vector / np.sum(vector)
    
        def softmax(vector):
            vec_ = []
            for vv in vector:
                vec_.append(np.exp(vv))
            vec_ = np.array(vec_)
            return vec_ / np.sum(vec_)

        value = []
        ct = self._ksize // 2
        for i in range(self._ksize):
            value.append(np.exp(- (i - ct)**2 / (2*self._sigma**2)))
        return eval(self._method)(value)
    
    def smooth_single(self, vector):
        vec_smooth = np.convolve(vector, self._kernel, mode='same')
        return vec_smooth

    def smooth(self, data):
        T, N, C = data.shape
        data_out = np.zeros_like(data)
        for i in range(N):
            for j in range(C):
                data_out[:, i, j] = self.smooth_single(data[:, i, j])
        return data_out.copy()


def process(args):
    # load data
    with open(args.data, "rb") as handle:
        skeleton_ori = pickle.load(handle)["skeleton"]
        handle.close()
    skeleton_smoothed = dict()
    print("The length of input data is %d" % len(skeleton_ori))
    key_list = []
    for skey in skeleton_ori:
        key_list.append(skey)
    num_threads = args.num_thread
    len_keys = len(key_list)
    seg_len = int(np.ceil(len_keys / num_threads))
    threads_list = []
    for i in range(num_threads):
        thread_ins = SmoothThread(
            keys=key_list[i*seg_len:min((i+1)*seg_len, len_keys)],
            sigma=args.sigma,
            method=args.method,
            ksize=args.ksize,
            data_in=skeleton_ori,
            data_out=skeleton_smoothed)
        thread_ins.start()
        threads_list.append(thread_ins)
    for thread_ins in threads_list:
        thread_ins.join()
    print("The length of output data is %d" % len(skeleton_smoothed))
    with open(args.output, "wb") as handle:
        pickle.dump({"skeleton": skeleton_smoothed}, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print_arguments(args)
    process(args)


