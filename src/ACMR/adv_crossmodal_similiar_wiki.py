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
from src.utils.retrieval import img2shape, shape2img

class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)

        self.epoch = 10000
        self.n_save_epoch = 10
        self.n_max_save = 100

        self.margin = .01
        self.alpha = 5
        self.batch_size = 64
        self.n_class = 10
        self.visual_feat_dim = 4096
        #self.word_vec_dim = 300
        self.word_vec_dim = 128
        self.lr_total = 0.0001
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.top_k = 50
        self.semantic_emb_dim = 40
        self.dataset_name = 'imagenet+shapenet'
        self.model_name = 'ACMR.ckpt'
        self.model_dir = 'ACMR_%d_%d_%d_similiar' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

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

        self.tar_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.tar_shape = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.pos_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.neg_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.pos_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.neg_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [None, self.model_params.n_class])
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size,1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.tar_img)
        self.emb_w = self.label_embed(self.tar_shape)
        self.emb_v_pos = self.visual_feature_embed(self.pos_img,reuse=True)
        self.emb_v_neg = self.visual_feature_embed(self.neg_img,reuse=True)
        self.emb_w_pos = self.label_embed(self.pos_txt,reuse=True)
        self.emb_w_neg = self.label_embed(self.neg_txt,reuse=True)

        # triplet loss
        margin = self.model_params.margin
        alpha = self.model_params.alpha
        v_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_v-self.emb_w_pos))
        v_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_v-self.emb_w_neg))
        w_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_w-self.emb_v_pos))
        w_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_w-self.emb_v_neg))
        # self.triplet_loss = tf.maximum(0.,margin+alpha*v_loss_pos-v_loss_neg) + tf.maximum(0.,margin+alpha*w_loss_pos-w_loss_neg)
        self.similar_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        self.similar_loss = tf.reduce_mean(self.similar_loss)
        tf.summary.scalar("similar_loss", self.similar_loss)

        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
                          tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        tf.summary.scalar('label_loss', self.label_loss)
        self.label_img_pred = tf.argmax(logits_v, 1)
        self.label_img_acc = tf.reduce_mean(tf.cast(tf.equal(self.label_img_pred, tf.argmax(self.y, 1)), tf.float32))
        tf.summary.scalar('label_img_acc', self.label_img_acc)
        self.label_shape_pred = tf.argmax(logits_w, 1)
        self.label_shape_acc = tf.reduce_mean(tf.cast(tf.equal(self.label_shape_pred, tf.argmax(self.y, 1)), tf.float32))
        self.label_class_acc =  tf.divide(tf.add(self.label_img_acc, self.label_shape_acc), 2.0)
        tf.summary.scalar('label_acc', self.label_class_acc)
        # TODO  triplet loss
        self.emb_loss = 100*self.label_loss + self.similar_loss
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([tf.shape(self.tar_img)[0], 1]),
                              tf.zeros([tf.shape(self.tar_img)[0], 1])], 1)
        all_emb_w = tf.concat([tf.zeros([tf.shape(self.tar_shape)[0], 1]),
                               tf.ones([tf.shape(self.tar_img)[0], 1])], 1)
        # all_emb_w = tf.
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)
        tf.summary.scalar('domain_loss', self.domain_class_loss)
        self.domain_img_class_acc = tf.equal(tf.greater(0.5, self.emb_v_class), tf.greater(0.5, all_emb_w))
        tf.summary.scalar('domain_img_acc', tf.reduce_mean(tf.cast(self.domain_img_class_acc, tf.float32)))
        self.domain_shape_class_acc = tf.equal(tf.greater(self.emb_w_class, 0.5), tf.greater(all_emb_v, 0.5))
        tf.summary.scalar('domain_shape_acc', tf.reduce_mean(tf.cast(self.domain_shape_class_acc, tf.float32)))
        self.domain_class_acc = tf.reduce_mean(tf.cast(tf.concat([self.domain_img_class_acc, self.domain_shape_class_acc], axis=0), tf.float32))
        tf.summary.scalar('domain_acc', self.domain_class_acc)

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
            # net = tf.nn.tanh(slim.fully_connected(L, 400, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net 
    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, self.model_params.n_class, scope='lc_fc_0')
        return net         
    def domain_classifier(self, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, self.model_params.semantic_emb_dim/2, scope='dc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim/4, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def train(self, sess):
        self.data = pairwise_input_data.read_data(config.SHAPE_TRAIN_FEATURE_FILE, config.IMG_TRAIN_FEATURE_FILE, config.TRAIN_LABEL_FILE,
                                               use_onehot=False, label_dim=self.model_params.n_class)
        # self.check_dirs()
        total_loss = self.emb_loss + self.domain_class_loss
        total_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_total,
            beta1=0.5).minimize(total_loss)
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars+self.vf_vars)
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config.MODEL_RESULT_DIR + '/train', sess.graph)
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(max_to_keep=self.model_params.n_max_save)

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
                # TODO triplet loss
                # batch_labels_ = batch_labels - np.ones_like(batch_labels)
                # label_binarizer = sklearn.preprocessing.LabelBinarizer()
                # label_binarizer.fit(range(max(batch_labels_)+1))
                # b = label_binarizer.transform(batch_labels_)
                b = pairwise_input_data.onehot(batch_labels, dim=self.model_params.n_class)
                # adj_mat = np.dot(b,np.transpose(b))
                # mask_mat = np.ones_like(adj_mat) - adj_mat
                # img_sim_mat = mask_mat*cosine_similarity(batch_feat, batch_feat)
                # txt_sim_mat = mask_mat*cosine_similarity(batch_vec, batch_vec)
                # img_neg_txt_idx = np.argmax(img_sim_mat,axis=1).astype(int)
                # txt_neg_img_idx = np.argmax(txt_sim_mat,axis=1).astype(int)
                # print(img_neg_txt_idx)
                # print(txt_neg_img_idx)
                # batch_vec_ = np.array(batch_vec)
                # batch_feat_ = np.array(batch_feat)
                # img_neg_txt = batch_vec_[img_neg_txt_idx,:]
                # txt_neg_img = batch_feat_[txt_neg_img_idx,:]
                img_neg_txt = self.find_neg_pair(batch_feat, batch_vec)
                txt_neg_img = self.find_neg_pair(batch_vec, batch_feat)
                summary, _, _, label_loss_val, sim_loss_val, emb_loss_val, domain_loss_val, domain_class_acc_val, label_class_acc_val = sess.run([merged, emb_train_op, domain_train_op, self.label_loss, self.similar_loss, self.emb_loss, self.domain_class_loss, self.domain_class_acc, self.label_class_acc],
                          feed_dict={self.tar_img: batch_feat,
                          self.tar_shape: batch_vec,
                          self.pos_txt: batch_vec,
                          self.neg_txt: img_neg_txt,
                          self.pos_img: batch_feat,
                          self.neg_img: txt_neg_img,
                          self.y: b,
                          self.y_single: np.transpose([batch_labels]),
                          self.l: l})
                print('Epoch=%d,batch=%d,time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, triplet_loss: %.8f, domain_class_acc: %.8f, label_class_acc: %.8f' %(
                    epoch+1, batch, time.time() - start_time, emb_loss_val, domain_loss_val, label_loss_val, sim_loss_val, domain_class_acc_val, label_class_acc_val
                ))
                train_writer.add_summary(summary, epoch * self.model_params.batch_size + batch)
                batch += 1
            if epoch % self.model_params.n_save_epoch == 0:
                self.save(epoch, sess)

    def find_neg_pair(self, fcs1, fcs2):
        """
        find negative pair for each value of fcs1 from fcs2
        :param fcs1:
        :param fcs2:
        :return:
        """
        assert fcs1.shape[0] == fcs2.shape[0]
        size = fcs1.shape[0]
        sims = cosine_similarity(fcs1, fcs1)
        result = []
        for i in range(size):
            sims[i][i] = -1.0
            neg_index = np.argmax(sims[i, :], axis=0).astype(int)
            result.append(neg_index)
        return fcs2[result]

    def test_find_neg_pair(self):
        fcs1, fcs2 = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]]), np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        result = self.find_neg_pair(fcs1, fcs2)
        assert result == [[3, 3, 3], [3, 3, 3], [1, 1, 1]]

    def eval(self, sess):
        self.img_data = pairwise_input_data.read_data(None, None, None, config.SHAPE_TEST_FEATURE_FILE, config.IMG_TEST_FEATURE_FILE, config.TEST_LABEL_FILE,
                                                      use_onehot=True, label_dim=self.model_params.n_class)
        self.shape_data = self.img_data
        # self.shape_data = pairwise_input_data.read_data(None, None, None, config.SHAPE_TEST_FEATURE_FILE, config.IMG_TEST_FEATURE_FILE, config.TEST_LABEL_FILE,
        #                                                 use_onehot=True, label_dim=self.model_params.n_class)
        start = time.time()
        self.saver = tf.train.Saver()
        self.load(sess)

        eval_shape_feature, eval_img_feature, eval_img_label = self.img_data.test.next_batch(300, shuffle=False)
        eval_joint_img_feature, eval_img_pred_label, eval_img_label_acc_val = sess.run([self.emb_v, self.label_img_pred, self.label_img_acc],
                                                                                       feed_dict={self.tar_img:eval_img_feature, self.y:eval_img_label})
        np.save(config.EXPERIMENTS_EVAL_IMG_FEATURE_FILE, eval_joint_img_feature)
        print("Joint image feature saved to %s" % config.IMG_TEST_FEATURE_FILE)
        print("Image prediction label:", eval_img_pred_label)
        # print("Image groun truth label:", eval_img_label)
        print("Image classification accuracy:%f"%eval_img_label_acc_val)

        eval_shape_feature, eval_img_feature, eval_shape_label = self.shape_data.test.next_batch(self.shape_data.test.size(), shuffle=False)
        eval_joint_shape_feature, eval_shape_pred_label, eval_shape_label_acc_val = sess.run([self.emb_w, self.label_shape_pred, self.label_shape_acc],
                                                                                             feed_dict={self.tar_shape:eval_shape_feature, self.y:eval_shape_label})
        np.save(config.EXPERIMENTS_EVAL_SHAPE_FEATURE_FILE, eval_joint_shape_feature)
        print("Shape predicition label:", eval_shape_pred_label)
        print("Shape classification accuracy:%f"%eval_shape_label_acc_val)
        print("Joint shape feature saved to %s" % config.SHAPE_TEST_FEATURE_FILE)

        print('[Eval] finished precision-scope in %4.4fs' % (time.time() - start))

        test_shape_vecs_trans, test_img_feats_trans, test_labels = eval_joint_shape_feature, eval_joint_img_feature, eval_shape_label
        # Do cross-modal class retrieval
        top_k = self.model_params.top_k
        avg_precs = []
        all_precs = []
        for k in range(1, top_k + 1):
            for i in range(len(test_shape_vecs_trans)):
                query_label = test_labels[i]

                # distances and sort by distances
                wv = test_shape_vecs_trans[i]
                diffs = test_img_feats_trans - wv
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k do top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0: topk]
                    if np.sum(query_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        # print("retrievaled label:", retrieved_label)
                        if np.sum(retrieved_label) == query_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        print('[Eval - shape2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))
        print(all_precs)
        with open(os.path.join(config.root_dir, 'data/result/shape2img_all_precision.pkl'), 'wb') as f:
            cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)

        avg_precs = []
        all_precs = []
        top_k = self.model_params.top_k
        for k in range(1, top_k + 1):
            for i in range(len(test_img_feats_trans)):
                query_img_feat = test_img_feats_trans[i]
                ground_truth_label = test_labels[i]

                # calculate distance and sort
                diffs = test_shape_vecs_trans - query_img_feat
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k in top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0: topk]
                    if np.sum(ground_truth_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(ground_truth_label) == retrieved_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        print('[Eval - img2shape] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))

        img2shape_pair_acc = img2shape(eval_joint_img_feature, eval_joint_shape_feature,
                                       np.arange(0, eval_joint_img_feature.shape[0]), top_k=self.model_params.top_k,
                                       tag="acmr-triplet", save_dir=config.EXPERIMENTS_EVAL_RESULT_DIR)
        print('[Test - img2shape pair:]', img2shape_pair_acc)
        print('[Eval] finished precision-scope in %4.4fs' % (time.time() - start))