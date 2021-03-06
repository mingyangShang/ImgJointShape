import numpy as np
import os
import random
import src.config as config
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSet(object):
    def __init__(self, fcs, labels):
        self._fcs, self._labels = fcs, labels
        self._epochs_completed, self._index_in_epoch = 0, 0
        self._num_examples = fcs.shape[0]

    @property
    def labels(self):
        return self._labels
    @property
    def fcs(self):
        return self._fcs
    @property
    def fc_dim(self):
        return self._fcs.shape[-1]

    def size(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._fcs = self._fcs[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            fcs_rest_part = self._fcs[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._fcs = self._fcs[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            fcs_new_part = self._fcs[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((fcs_rest_part, fcs_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._fcs[start:end], self._labels[start:end]


def read_data(train_feature_file=None, train_label_file=None, eval_feature_file=None, eval_label_file=None, use_onehot=True, label_dim=10):
    train_feature = np.load(train_feature_file) if train_feature_file else None
    train_label = np.load(train_label_file) if train_label_file else None
    eval_feature = np.load(eval_feature_file) if eval_feature_file else None
    eval_label = np.load(eval_label_file) if eval_label_file else None
    if use_onehot and train_label_file:
        train_label = onehot(train_label, label_dim)
    if use_onehot and eval_label_file:
        eval_label = onehot(eval_label, label_dim)
    train_dataset = DataSet(train_feature, train_label) if train_label_file else None
    test_dataset = DataSet(eval_feature, eval_label) if eval_label_file else None
    return base.Datasets(train=train_dataset, test=test_dataset, validation=None)

def onehot(labels, dim):
    label_count = np.shape(labels)[0]
    labels2 = np.zeros(shape=[label_count, dim]) # TODO shape=[batch_size, n_classes]
    labels2[np.arange(label_count), labels] = 1
    return labels2

if __name__ == '__main__':
    imgs_data = read_data(config.SHAPE_TRAIN_FEATURE_FILE, config.SHAPE_TRAIN_LABEL_FILE)
    print("train data size: ", imgs_data.train.size())
    # print("img train data next batch 10:", imgs_data.train.next_batch(10, shuffle=True))
    print("feature dim:", imgs_data.train.fc_dim)