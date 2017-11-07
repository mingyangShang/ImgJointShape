import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def pca_reduction(feature, to_dim, save_path=''):
    print("feature reshaping")
    feature = np.reshape(feature, newshape=[-1, feature.shape[-1]])
    pca = PCA(n_components=to_dim)
    print("fitting")
    pca.fit(feature)
    print("pca transforming")
    reducted_feature = pca.transform(feature)
    if len(save_path):
        print("saving")
        np.save(save_path, reducted_feature)
        print("reducted feature saved to %s"%save_path)
    return reducted_feature

def pca_vis(features, labels, tag="pca", fig_index=1):
    fig = plt.figure(fig_index)
    ax = fig.add_subplot(111)
    ax.set_title(tag)
    plt.xlabel('X')
    plt.ylabel('Y')
    features = features.astype(int)
    ax.scatter(features[:, 0], features[:, 1], c=labels, marker='.')
    plt.legend("shapenet-shape")

def tsne_reduction(feature, to_dim, save_path=''):
    print("feature reshaping")
    feature = np.reshape(feature, newshape=[-1, feature.shape[-1]])
    print("tsne fitting and transforming")
    reducted_feature = TSNE(n_components=to_dim).fit_transform(feature)
    if len(save_path):
        print("saving")
        np.save(save_path, reducted_feature)
        print("reducted feature saved to %s"%save_path)
    return reducted_feature

def tsne_visualization(pre_img_feature_file, pre_shape_feature_file, to_dim=2):
    tsne_reduction(np.load(pre_img_feature_file), to_dim=to_dim, save_path=pre_img_feature_file[:pre_img_feature_file.index(".npy")]+"_tsne_2dim.npy")
    tsne_reduction(np.load(pre_shape_feature_file), to_dim=to_dim, save_path=pre_shape_feature_file[:pre_shape_feature_file.index(".npy")]+"_tsne_2dim.npy")

if __name__ == "__main__":
    pca_reduction(np.load("/home1/shangmingyang/projects/tensorflow-vgg/feature/imagenet_class_center_feature.npy"), 2, \
                  save_path='/home1/shangmingyang/data/ImgJoint3D/feature/imagenet_class_center_feature_pca_2dim.npy')
    tsne_reduction(np.load("/home1/shangmingyang/projects/tensorflow-vgg/feature/imagenet_class_center_feature.npy"), 2, \
                  save_path='/home1/shangmingyang/data/ImgJoint3D/feature/imagenet_class_center_feature_tsne_2dim.npy')
    # tsne_visualization("/home1/shangmingyang/data/ImgJoint3D/feature/train_imagenet_feature.npy", "/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature.npy")
    # tsne_visualization("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature_all.npy", "/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy")
    # pca_reduction(np.load(src.config.IMG_FEATURE_FILE), src.config.IMG_FEATURE_DIM, save_path=src.config.REDUCTED_IMG_FEATURE_FILE)
    # pca_reduction(np.load(src.config.IMG_FEATURE_FILE), src.config.IMG_FEATURE_DIM, save_path=src.config.REDUCTED_IMG_FEATURE_FILE)
    # pca_vis(np.load('/home/shangmingyang/Desktop/shapenet55_nocolor_2dim.npy'), np.load('/home/shangmingyang/Desktop/all_shapenet55_labels.npy'), tag="all_shapenet55_shape", fig_index=1)
    # pca_vis(np.load('/home/shangmingyang/Desktop/train_img_feature_all_2dim.npy')[::12], np.load('/home/shangmingyang/Desktop/all_shapenet55_labels.npy'), tag="all_shapenet55_img", fig_index=2)
    # pca_vis(np.load('/home/shangmingyang/Desktop/train_shape_feature_2dim.npy'),
    #         np.load('/home/shangmingyang/Desktop/train_shape_labels.npy'), tag="train_shapenet55_shape", fig_index=1)
    # pca_vis(np.load('/home/shangmingyang/Desktop/train_imagenet_feature_2dim.npy'),
    #         np.load('/home/shangmingyang/Desktop/train_imagenet_labels.npy'), tag="train_imagenet_img", fig_index=2)
    # plt.show()
    # pca_reduction(np.load('/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature.npy'), 2, save_path='/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature_2dim.npy')
    # pca_reduction(np.load('/home1/shangmingyang/data/ImgJoint3D/feature/train_imagenet_feature.npy'), 2, save_path='/home1/shangmingyang/data/ImgJoint3D/feature/train_imagenet_feature_2dim.npy')
