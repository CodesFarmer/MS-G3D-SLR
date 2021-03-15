import argparse
import pickle
import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


def softmax(scores):
    scores_exp = np.exp(scores)
    return scores_exp / np.sum(scores_exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-list',
                        required=True,
                        help='file contain all files to be merged')
    parser.add_argument('--ground-truth',
                        required=True,
                        help='ground truth to the files')
    parser.add_argument('--output',
                        required=True,
                        help='ground truth to the files')
    parser.add_argument('--method',
                        required=True,
                        help='ground truth to the files')
    arg = parser.parse_args()

    file_list = None
    with open(arg.result_list, 'r') as handle:
        file_list = handle.readlines()
        handle.close()
    results = []
    wnorm = 0.0
    for line in file_list:
        file_path, file_weights = line.split(' ')
        with open(file_path.strip(), 'rb') as handle:
            predictions = list(pickle.load(handle).items())
            results.append((predictions, float(file_weights)))
            wnorm += float(file_weights)
            handle.close()

    with open(arg.ground_truth, 'rb') as handle:
        label = np.array(pickle.load(handle))
        handle.close()

    right_num = total_num = right_num_5 = 0
    merged_score = OrderedDict()
    for i in tqdm(range(len(label[0]))):
        sid_1, l = label[:, i]
        sid, r_merge = results[0][0][i]
        if arg.method == "softmax":
            r_merge = softmax(r_merge) * results[0][1] / wnorm
        elif arg.method == "sum":
            r_merge = r_merge * results[0][1] / wnorm
        else:
            raise "Does not contain the method %s" % (arg.method)
        # r_merge = r_merge * results[0][1] / wnorm
        for res in results[1:]:
            sid_cur, r_cur = res[0][i]
            assert sid_1 == sid and sid == sid_cur
            if arg.method == "softmax":
                r_merge = r_merge + softmax(r_cur) * res[1] / wnorm
            elif arg.method == "sum":
                r_merge = r_merge + r_cur * res[1] / wnorm
            else:
                raise "Does not contain the method %s" % (arg.method)
        merged_score[sid] = r_merge.copy()
        rank_5 = r_merge.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r_merge)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    with open(arg.output, 'wb') as handle:
        pickle.dump(merged_score, handle)
        handle.close()

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

