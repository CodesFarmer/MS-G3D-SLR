import os
import pickle
import argparse
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import itertools
import random
from multiprocessing import Process, Queue
import time


def findsubsets(S, m):
    return set(itertools.combinations(S, m))


def softmax(scores):
    scores_exp = np.exp(scores)
    return scores_exp / np.sum(scores_exp)


def generate_model_matrix(model_list, anno_label, num_classes=226):
    with open(anno_label, 'rb') as handle:
        anno = np.array(pickle.load(handle))
        handle.close()

    modename2id = dict()
    with open(model_list, "r") as handle:
        file_list = handle.readlines()
        handle.close()

    model_num = len(file_list)
    sample_num = len(anno[0])
    score_matrix = np.zeros(shape=(model_num, sample_num, num_classes), dtype=np.float)
    results = []
    wnorm = 0.0
    for i, line in enumerate(file_list):
        file_path = line.strip()
        modename2id[file_path] = i + 1
        with open(file_path, 'rb') as handle:
            predictions = list(pickle.load(handle).items())
            results.append(predictions)
            handle.close()
    right_num = total_num = right_num_5 = 0

    for i in range(model_num):
        pred_cur = results[i]
        for j in range(sample_num):
            sid_cur, label = anno[:, j]
            sid_pr, score_pr = pred_cur[j]
            assert sid_pr == sid_cur
            score_matrix[i, j, :] = softmax(score_pr)
    return score_matrix, modename2id, anno[1]


class Searcher(Process):
    def __init__(self, subsets, score_matrix, label, queue):
        super(Searcher, self).__init__()
        self._subsets = subsets
        self._queue = queue
        self._score_matrix = score_matrix
        self._stop = False
        self._is_stop = False
        self._label = np.array(label).astype(np.float32)

    def run(self):
        for subset in self._subsets:
            if self._stop:
                break
            score_matrix_ = self._score_matrix[subset, :, :]
            score_merge = np.sum(score_matrix_, axis=0)
            pr_label = np.argmax(score_merge, axis=1)
            precision = np.sum((np.array(pr_label)-self._label)==0) / len(self._label)
            subset_id = ""
            for samid in subset:
                subset_id += "%d_" % (samid+1)
            subset_id = subset_id[:-1]
            self._queue.put((subset_id, precision))
        self._is_stop = True

    def is_stop(self):
        return self._is_stop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-list',
                        type=str,
                        required=True,
                        help='file contain all files to be merged')
    parser.add_argument('--ground-truth',
                        type=str,
                        required=True,
                        help='ground truth to the files')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='output file to save all combination score')
    parser.add_argument('--max-num',
                        type=int,
                        default=4,
                        help='the max number of model to ensemble')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='the number of worker to run evaluation')
    arg = parser.parse_args()

    score_matrix, modename2id, label = generate_model_matrix(arg.result_list, arg.ground_truth)
    model_list = [i for i in range(score_matrix.shape[0])]
    possible_combination = []
    for i in range(3, arg.max_num+1):
        possible_subsets = findsubsets(model_list, i)
        for subset in possible_subsets:
            possible_combination.append(list(subset))
    random.shuffle(possible_combination)
    len_subset = len(possible_combination)
    seg_len = int(np.ceil(len_subset / arg.workers))
    processors = []
    queue = Queue()
    for i in range(arg.workers):
        processor = Searcher(
            subsets=possible_combination[i*seg_len:min((i+1)*seg_len, len_subset)],
            score_matrix=score_matrix,
            queue=queue,
            label=label)
        processor.start()
        processors.append(processor)

    handle = open(arg.output, "w", buffering=1)
    for kname in modename2id:
        handle.write("model:%s, %d\n" % (kname, modename2id[kname]))
    latest_time = time.time()
    last_time = time.time()
    while True:
        if not queue.empty():
            combination, precision = queue.get()
            handle.write("%.6f %s\n" % (precision, combination))
            handle.flush()
            last_time = time.time()
        latest_time = time.time()
        if latest_time - last_time > 5:
            handle.close()
            break
    for processor in processors:
        processor.join()



