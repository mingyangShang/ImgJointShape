import numpy as np
import os

from src.utils import visualization
from src.utils import reduction
import src.config as config

def joint_feature_modal_visualization(joint_img_fc_path, joint_shape_fc_path, fig_index=0):
    # tsne reduction
    img_fcs, shape_fcs = np.load(joint_img_fc_path), np.load(joint_shape_fc_path)
    # reducted_fc = reduction.tsne_reduction(np.concatenate([img_fcs, shape_fcs], axis=0), to_dim=2, save_path="../../data/result/visual/test_joint_tsne.npy")
    reducted_fc = np.load("../../data/result/visual/test_joint_tsne.npy")
    img_entire_labels, shape_entire_labels = np.zeros(img_fcs.shape[0]), np.ones(shape_fcs.shape[0])
    modal_labels = np.concatenate([img_entire_labels, shape_entire_labels], axis=0)
    visualization.feature_reduction_vis(reducted_fc, modal_labels, tag="modal visualization using tsne", fig_index=fig_index)

def joint_feature_class_visualization(joint_img_fc_path, joint_shape_fc_path, img_label_path, shape_label_path, fig_index=0):
    # img_fcs, shape_fcs = np.load(joint_img_fc_path), np.load(joint_shape_fc_path)
    # reducted_fc = reduction.tsne_reduction(np.concatenate([img_fcs, shape_fcs], axis=0), to_dim=2, save_path="../../data/result/visual/test_joint_tsne.npy")
    reducted_fc = np.load("../../data/result/visual/test_joint_tsne.npy")
    img_labels, shape_labels = np.load(img_label_path), np.load(shape_label_path)
    img_labels, shape_labels = np.repeat(img_labels, 12, axis=0), np.repeat(shape_labels, 12, axis=0)
    img_hint, shape_hint = [str(o) for o in np.arange(1, img_labels.shape[0]+1)], [str(o) for o in np.arange(1, shape_labels.shape[0]+1)]
    hint_text = np.concatenate([img_hint, shape_hint], axis=0)
    hint_text = None
    visualization.feature_reduction_vis(reducted_fc, np.concatenate([img_labels, shape_labels]), tag="class visualization using tsne", text=hint_text, fig_index=fig_index)

def pair_distance_compare(pair_dists_path, rank_indexs_path, label_path):
    pair_dists, rank_indexs, labels = np.load(pair_dists_path), np.load(rank_indexs_path), np.load(label_path)
    pair_dists, rank_indexs = pair_dists[np.arange(pair_dists.shape[-1]),:], rank_indexs[np.arange(rank_indexs.shape[-1]),:]
    print(labels.tolist())
    u_labels, i_labels, c_labels = np.unique(labels, return_index=True, return_counts=True)
    mean_pair_dists, mean_min_dists, mean_max_dists, mean_middle_dists, mean_all_dists = [], [], [], [], []
    var_pair_dists, var_min_dists, var_max_dists, var_middle_dists, var_all_dists = [], [], [], [], []
    for i in range(len(i_labels)):
        n_class_data = c_labels[i]
        class_dists = pair_dists[np.arange(i, i+n_class_data), :]
        class_dists = class_dists[: ,np.arange(i, i+n_class_data)]
        class_indexs = np.array([np.argsort(class_dists[j]) for j in range(class_dists.shape[0])])
        # class_indexs = rank_indexs[np.arange(i, i+n_class_data), :]
        # class_indexs = class_indexs[:, np.arange(i, i+n_class_data)]
        print(class_dists.shape, class_indexs.shape)
        mean_pair_dist, var_pair_dist = np.mean([class_dists[i][i] for i in range(n_class_data)]), np.var([class_dists[i][i] for i in range(n_class_data)])
        mean_min_dist, var_min_dist = np.mean(class_dists[np.arange(n_class_data), class_indexs[:, 0]]), np.var(class_dists[np.arange(n_class_data), class_indexs[:, 0]])
        mean_max_dist, var_max_dist = np.mean(class_dists[np.arange(n_class_data), class_indexs[:, -1]]), np.var(class_dists[np.arange(n_class_data), class_indexs[:, -1]])
        mean_middle_dist, var_middle_dist = np.mean(class_dists[np.arange(n_class_data), class_indexs[:, n_class_data/2]]), np.var(class_dists[np.arange(n_class_data), class_indexs[:, n_class_data/2]])
        mean_all_dist, var_all_dist = np.mean(class_dists), np.var(class_dists)

        mean_pair_dists.append(mean_pair_dist)
        mean_min_dists.append(mean_min_dist)
        mean_max_dists.append(mean_max_dist)
        mean_middle_dists.append(mean_middle_dist)
        mean_all_dists.append(mean_all_dist)

        var_pair_dists.append(var_pair_dist)
        var_min_dists.append(var_min_dist)
        var_max_dists.append(var_max_dist)
        var_middle_dists.append(var_middle_dist)
        var_all_dists.append(var_all_dist)

    print("mean_pair_dist=%f,mean_min_dist=%f,mean_max_dist=%f,mean_middle_dist=%f,mean_all_dist=%f"
          %(np.mean(mean_pair_dists), np.mean(mean_min_dists), np.mean(mean_max_dists), np.mean(mean_middle_dists), np.mean(mean_all_dists)))
    print("var_pair_dist=%f,var_min_dist=%f,var_max_dist=%f,var_middle_dist=%f,var_all_dist=%f"
          %(np.mean(var_pair_dists), np.mean(var_min_dists), np.mean(var_max_dists), np.mean(var_middle_dists), np.mean(var_all_dists)))

if __name__ == '__main__':
    # pair_distance_compare('data/shape2img_dists.npy', 'data/shape2img_indexs.npy', config.TEST_LABEL_FILE)
    joint_feature_modal_visualization('../../data/result/visual/image_eval_first_negativepair_modelnet10_acmr_joint_features.npy', '../../data/result/visual/shape_eval_first_negativepair_modelnet10_acmr_joint_features.npy', fig_index=0)
    joint_feature_class_visualization('../../data/result/visual/image_eval_first_negativepair_modelnet10_acmr_joint_features.npy', '../../data/result/visual/shape_eval_first_negativepair_modelnet10_acmr_joint_features.npy',
                                     '../../data/image/modelnet10/test_labels_modelnet10.npy', '../../data/image/modelnet10/test_labels_modelnet10.npy', fig_index=1)
    visualization.show()
