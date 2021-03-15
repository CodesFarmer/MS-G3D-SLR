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
        '--label',
        type=str,
        required=True,
        help='label file')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='the path to save the smoothed data')
    parser.add_argument(
        '--nclips',
        type=int,
        default=64,
        help='the number of threads')
    parser.add_argument(
        '--method',
        type=str,
        default="uniform",
        help='normalize for weights')
    parser.add_argument(
        '--num_thread',
        type=int,
        default=6,
        help='the number of threads')
    return parser


class Bottling(threading.Thread):
    def __init__(self, keys, method, clips, data_in, data_out, start_id):
        super(Bottling, self).__init__()
        self._keys = keys
        self._method = method
        self._clips = clips
        self._data_in = data_in
        self._data_out = data_out
        self._start_id = start_id

    def run(self):
        process = tqdm(self._keys, dynamic_ncols=True)
        for sid, skey in enumerate(process):
            if len(self._data_in[skey].shape) < 3:
                print(skey)
            self._data_out[self._start_id+sid, :, :, :, :] = self.bottle_up(self._data_in[skey])

    def  _search_num1_num2(self, total_frame, lower, upper):
        '''
        find the integer x1 x2 so that
        x1 + x2 = _clips
        x1 * lower + x2 * upper = total_frame
        '''
        for x1 in range(0, self._clips):
            x2 = self._clips - x1
            if x1 * lower + x2 * upper == total_frame:
                return x1, x2
        return 0, 0
    
    def bottle_up(self, data):
        T, N, C = data.shape
        data_out = np.zeros(shape=(C, self._clips, N, 1), dtype=np.float32)
        data = np.transpose(data, axes=[2, 0, 1])
        if T > self._clips:
            if self._method == "left":
                data_out[:, :, :, 0] = data[:, :self._clips, :]
            elif self._method == "right":
                data_out[:, :, :, 0] = data[:, -self._clips:, :]
            elif self._method == "middle":
                data_out[:, :, :, 0] = data[:, (T-self._clips)//2:(T-self._clips)//2+self._clips, :]
            elif self._method == "uniform":
                avg_interval = T / self._clips
                lower = np.floor(avg_interval).astype(np.int32)
                upper = np.ceil(avg_interval).astype(np.int32)
                num1, num2 = self._search_num1_num2(T, lower, upper)
                seg_len = [lower] * num1 + [upper] * num2
                np.random.shuffle(seg_len)
                act_seg_len = [0] + np.cumsum(seg_len).tolist()
                clip_offsets = [np.random.randint(act_seg_len[i], act_seg_len[i+1]) for i in range(self._clips)]
                data_out[:, :, :, 0] = data[:, clip_offsets, :]
        elif T <= self._clips:
            data_out[:, :T, :, 0] = data
        return data_out.copy()


def process(args):
    # load data
    with open(args.data, "rb") as handle:
        skeleton_ori = pickle.load(handle)["skeleton"]
        handle.close()
    with open(args.label, "rb") as handle:
        sample_name, sample_lbl = pickle.load(handle, encoding='latin1')
        handle.close()
    print("The length of input data is %d" % len(skeleton_ori))
    key_list = []
    for skey in sample_name:
        if "val" in args.label and "mini" not in args.label:
            align_name = "frames/val/%s" % skey.replace(".json", "")
        elif "train" in args.label or "mini" in args.label:
            align_name = "frames/train/%s" % skey.replace(".json", "")
        elif "test":
            align_name = "frames/test/%s" % skey.replace(".json", "")
        else:
            raise "Label must be one of val, train or test"
        key_list.append(align_name)
    num_threads = args.num_thread
    len_keys = len(key_list)
    seg_len = int(np.ceil(len_keys / num_threads))
    threads_list = []
    _, N, C = skeleton_ori[key_list[0]].shape
    skeleton_bottled = np.zeros(shape=(len_keys, C, args.nclips, N, 1))
    for i in range(num_threads):
        thread_ins = Bottling(
            keys=key_list[i*seg_len:min((i+1)*seg_len, len_keys)],
            method=args.method,
            clips=args.nclips,
            start_id=i*seg_len,
            data_in=skeleton_ori,
            data_out=skeleton_bottled)
        thread_ins.start()
        threads_list.append(thread_ins)
    for thread_ins in threads_list:
        thread_ins.join()
    print("The length of output data is ", skeleton_bottled.shape)
    with open(args.output, "wb") as handle:
        np.save(handle, skeleton_bottled)
        handle.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print_arguments(args)
    process(args)


