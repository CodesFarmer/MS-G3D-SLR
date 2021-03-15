import pickle
import numpy as np


file_mini_train_data = "hand/minitrain_data_joint.npy"
file_mini_val_data = "hand/minival_data_joint.npy"
file_full_val_data = "hand/val_data_joint.npy"
file_mini_train_label = "hand/minitrain_label.pkl"
file_mini_val_label = "hand/minival_label.pkl"
file_full_val_label = "hand/val_label.pkl"


def load_label(filename):
    with open(filename, "rb") as handle:
        samplename, label = pickle.load(handle)
        handle.close()
    return samplename, label


def load_data(filename):
    with open(filename, "rb") as handle:
        data = np.load(handle)
        handle.close()
    return data


mt_sample, mt_label = load_label(file_mini_train_label)
mv_sample, mv_label = load_label(file_mini_val_label)
fv_sample, fv_label = load_label(file_full_val_label)
mt_data = load_data(file_mini_train_data)
mv_data = load_data(file_mini_val_data)
fv_data = load_data(file_full_val_data)

points_data = np.concatenate([mt_data, mv_data, fv_data], axis=0)
sample_list = mt_sample + mv_sample + fv_sample
sample_label = mt_label + mv_label + fv_label

with open("hand/trainval_data_joint.npy", "wb") as handle:
    np.save(handle, points_data)
    handle.close()
with open("hand/trainval_label.pkl", "wb") as handle:
    pickle.dump((sample_list, sample_label), handle, pickle.HIGHEST_PROTOCOL)
    handle.close()



