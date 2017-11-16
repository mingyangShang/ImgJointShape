from __future__ import print_function
import os, time, cPickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
import sklearn.preprocessing
from base_model import BaseModel, BaseModelParams, BaseDataIter
from flip_gradient import flip_gradient
from sklearn.metrics.pairwise import cosine_similarity

from src.data import pairwise_input_data
import src.config as config

class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open('./data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
            self.train_img_feats = cPickle.load(f)
        with open('./data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
            self.train_txt_vecs = cPickle.load(f)
        with open('./data/wikipedia_dataset/train_labels.pkl', 'rb') as f:
            self.train_labels = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
            self.test_img_feats = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
            self.test_txt_vecs = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as f:
            self.test_labels = cPickle.load(f)

        self.num_train_batch = len(self.train_img_feats) / self.batch_size
        self.num_test_batch = len(self.test_img_feats) / self.batch_size

    def train_data(self):
        for i in range(self.num_train_batch):
            batch_img_feats = self.train_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_txt_vecs = self.train_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.train_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_feats = self.test_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_txt_vecs = self.test_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.test_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i


class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)

        self.epoch = 10000
        self.n_save_epoch = 10
        self.n_max_save = 100
        self.n_class = 3
        self.margin = .1
        self.alpha = 5
        self.batch_size = 64
        self.visual_feat_dim = 4096
        # self.word_vec_dim = 300
        self.word_vec_dim = 5000
        self.lr_total = 0.0001
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.top_k = 50
        self.semantic_emb_dim = 40
        self.dataset_name = 'imagenet+shapenet'
        self.model_name = 'ACMR.ckpt'
        self.model_dir = 'ACMR_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        self.checkpoint_dir = '/home1/shangmingyang/projects/ImgJointShape/model'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        self.tar_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.tar_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.pos_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.neg_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.pos_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.neg_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [None, self.model_params.n_class])
        self.y_single = tf.placeholder(tf.int32, [None, 1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.tar_img)
        self.emb_w = self.label_embed(self.tar_txt)
        self.emb_v_pos = self.visual_feature_embed(self.pos_img, reuse=True)
        self.emb_v_neg = self.visual_feature_embed(self.neg_img, reuse=True)
        self.emb_w_pos = self.label_embed(self.pos_txt, reuse=True)
        self.emb_w_neg = self.label_embed(self.neg_txt, reuse=True)

        # triplet loss
        margin = self.model_params.margin
        alpha = self.model_params.alpha
        v_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_v - self.emb_w_pos))
        v_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_v - self.emb_w_neg))
        w_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_w - self.emb_v_pos))
        w_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_w - self.emb_v_neg))
        self.triplet_loss = tf.maximum(0., margin + alpha * v_loss_pos - v_loss_neg) + tf.maximum(0.,
                                                                                                  margin + alpha * w_loss_pos - w_loss_neg)

        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
                          tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.emb_loss = 100 * self.label_loss + self.triplet_loss
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                               tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                               tf.ones([self.model_params.batch_size, 1])], 1)
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
                                 tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)

        self.t_vars = tf.trainable_variables()
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]

    def visual_feature_embed(self, X, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X, 512, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, self.model_params.semantic_emb_dim, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net

    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 10, scope='lc_fc_0')
        return net

    def domain_classifier(self, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, self.model_params.semantic_emb_dim / 2, scope='dc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim / 4, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def train(self, sess):
        # self.check_dirs()
        self.data = pairwise_input_data.read_data(config.SHAPE_TRAIN_FEATURE_FILE, config.IMG_TRAIN_FEATURE_FILE,
                                                  config.SHAPE_TRAIN_LABEL_FILE,                                   use_onehot=False)
        total_loss = self.emb_loss + self.domain_class_loss
        total_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_total,
            beta1=0.5).minimize(total_loss)
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars + self.vf_vars)
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

        start_time = time.time()
        map_avg_ti = []
        map_avg_it = []
        adv_loss = []
        emb_loss = []
        for epoch in range(self.model_params.epoch):
            batch = 1
            p = float(epoch) / self.model_params.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            while batch * self.model_params.batch_size <= self.data.train.size():
                # create one-hot labels
                batch_vec, batch_feat, batch_labels = self.data.train.next_batch(self.model_params.batch_size)
                p = float(epoch) / self.model_params.epoch
                l = 2. / (1. + np.exp(-10. * p)) - 1
                # create one-hot labels
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(max(batch_labels_) + 1))
                b = label_binarizer.transform(batch_labels_)
                adj_mat = np.dot(b, np.transpose(b))
                mask_mat = np.ones_like(adj_mat) - adj_mat
                img_sim_mat = mask_mat * cosine_similarity(batch_feat, batch_feat)
                txt_sim_mat = mask_mat * cosine_similarity(batch_vec, batch_vec)
                img_neg_txt_idx = np.argmax(img_sim_mat, axis=1).astype(int)
                txt_neg_img_idx = np.argmax(txt_sim_mat, axis=1).astype(int)
                batch_vec_ = np.array(batch_vec)
                batch_feat_ = np.array(batch_feat)
                img_neg_txt = batch_vec_[img_neg_txt_idx, :]
                txt_neg_img = batch_feat_[txt_neg_img_idx, :]
                sess.run([emb_train_op, domain_train_op],
                         feed_dict={self.tar_img: batch_feat,
                                    self.tar_txt: batch_vec,
                                    self.pos_txt: batch_vec,
                                    self.neg_txt: img_neg_txt,
                                    self.pos_img: batch_feat,
                                    self.neg_img: txt_neg_img,
                                    self.y: batch_labels,
                                    self.y_single: np.transpose([batch_labels]),
                                    self.l: l})
                label_loss_val, triplet_loss_val, emb_loss_val, domain_loss_val = sess.run(
                    [self.label_loss, self.triplet_loss, self.emb_loss, self.domain_class_loss],
                    feed_dict={self.tar_img: batch_feat,
                               self.tar_txt: batch_vec,
                               self.pos_txt: batch_vec,
                               self.neg_txt: img_neg_txt,
                               self.pos_img: batch_feat,
                               self.neg_img: txt_neg_img,
                               self.y: b,
                               self.y_single: np.transpose([batch_labels]),
                               self.l: l})
                print(
                    'Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, triplet_loss: %.8f' % (
                        epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, emb_loss_val,
                        domain_loss_val, label_loss_val, triplet_loss_val
                    ))
                # if epoch == (self.model_params.epoch - 1):
                #    self.emb_v_eval, self.emb_w_eval = sess.run([self.emb_v, self.emb_w],
                #             feed_dict={
                #                 self.tar_img: batch_feat,
                #                 self.tar_txt: batch_vec,
                #                 self.y: b,
                #                 self.y_single: np.transpose([batch_labels]),
                #                 self.l: l})
                #    with open('./data/wikipedia_dataset/train_img_emb.pkl', 'wb') as f:
                #        cPickle.dump(self.emb_v_eval, f, cPickle.HIGHEST_PROTOCOL)
                #    with open('./data/wikipedia_dataset/train_txt_emb.pkl', 'wb') as f:
                #        cPickle.dump(self.emb_w_eval, f, cPickle.HIGHEST_PROTOCOL)