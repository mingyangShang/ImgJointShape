import numpy as np
import os
import random
import src.config as config
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSet(object):
    def __init__(self, shape_fcs, img_fcs, labels):
        self._shape_fcs, self._img_fcs, self._labels = shape_fcs, img_fcs, labels
        self._epochs_completed, self._index_in_epoch = 0, 0
        self._num_examples = img_fcs.shape[0]
        self.n_each_pair = img_fcs.shape[1]

    @property
    def labels(self):
        return self._labels
    @property
    def fcs(self):
        return self._shape_fcs
    @property
    def fc_dim(self):
        return self._shape_fcs.shape[-1]

    def size(self):
        return self._shape_fcs.shape[0]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._img_fcs = self._img_fcs[perm0]
            self._shape_fcs = self._shape_fcs[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            shape_fcs_rest_part = self._shape_fcs[start:self._num_examples]
            img_fcs_rest_part = self._img_fcs[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._shape_fcs = self._shape_fcs[perm]
                self._img_fcs = self._img_fcs[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            fcs_new_part = self._shape_fcs[start:end]
            img_fcs_new_part = self._img_fcs[start:end]
            labels_new_part = self._labels[start:end]
            result = np.concatenate((shape_fcs_rest_part, fcs_new_part), axis=0), np.concatenate((img_fcs_rest_part, img_fcs_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            result = self._shape_fcs[start:end], self._img_fcs[start:end], self._labels[start:end]
        # return result
        return np.repeat(result[0], self.n_each_pair, axis=0), np.reshape(result[1], [-1, result[1].shape[-1]]), np.repeat(result[2], self.n_each_pair, axis=0)

def read_data(train_shape_feature_file=None, train_img_feature_file=None, train_shape_label_file=None, eval_shape_feature_file=None, eval_img_feature_file=None, eval_shape_label_file=None, use_onehot=True, label_dim=10, views_th=1):
    print(train_img_feature_file, train_shape_feature_file)
    train_shape_feature = np.load(train_shape_feature_file) if train_shape_feature_file else None

    # train_shape_feature = np.repeat(train_shape_feature, 2, axis=0) if train_shape_feature_file else None
    train_img_feature = np.load(train_img_feature_file) if train_img_feature_file else None
    # train_img_feature = extract_first_n_views(train_img_feature)
    train_shape_label = np.load(train_shape_label_file).astype(int) if train_shape_label_file else None

    eval_shape_feature = np.load(eval_shape_feature_file) if eval_shape_feature_file else None
    eval_img_feature = np.load(eval_img_feature_file) if eval_img_feature_file else None
    # eval_img_feature = extract_first_n_views(eval_img_feature)
    eval_shape_label = np.load(eval_shape_label_file).astype(int) if eval_shape_label_file else None
    if use_onehot and train_shape_label_file:
        train_shape_label = onehot(train_shape_label, label_dim)
    if use_onehot and eval_shape_label_file:
        eval_shape_label = onehot(eval_shape_label, label_dim)
    train_dataset = DataSet(train_shape_feature, train_img_feature, train_shape_label) if train_shape_label_file else None
    test_dataset = DataSet(eval_shape_feature, eval_img_feature, eval_shape_label) if eval_shape_label_file else None
    return base.Datasets(train=train_dataset, test=test_dataset, validation=None)

def extract_first_n_views(fcs, n_first_views=1):
    fcs = fcs[:, np.arange(0, n_first_views), :] if fcs is not None else None
    fcs = np.reshape(fcs, [-1, fcs.shape[-1]]) if fcs is not None else None
    return fcs

def onehot(labels, dim):
    label_count = np.shape(labels)[0]
    labels2 = np.zeros(shape=[label_count, dim], dtype=int) # TODO shape=[batch_size, n_classes]
    labels2[np.arange(label_count), labels] = 1
    return labels2

if __name__ == '__main__':
    imgs_data = read_data(config.SHAPE_TRAIN_FEATURE_FILE, config.IMG_TRAIN_FEATURE_FILE, config.SHAPE_TRAIN_LABEL_FILE, label_dim=10)
    shape, img, labels = imgs_data.train.next_batch(10)
    print(shape.shape, img.shape, labels.shape)
    print(labels)
    print("train data size: ", imgs_data.train.size())
    # print("img train data next batch 10:", imgs_data.train.next_batch(10, shuffle=True))
    print("feature dim:", imgs_data.train.fc_dim)