import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import  TSNE

def pca_reduction(feature, to_dim, save_path=''):
    print("feature reshaping")
    feature = np.reshape(feature, newshape=[-1, feature.shape[-1]])
    pca = PCA(n_components=to_dim)
    print("fitting")
    pca.fit(feature)
    print("transforming")
    reducted_feature = pca.transform(feature)
    if len(save_path):
        print("saving")
        np.save(save_path, reducted_feature)
        print("reducted feature saved to %s"%save_path)
    return reducted_feature

def tsne_reduction(feature, to_dim, save_path=''):
    print("feature reshaping")
    feature = np.reshape(feature, newshape=[-1, feature.shape[-1]])
    print("fitting and transforming")
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
    # tsne_visualization("/home1/shangmingyang/data/ImgJoint3D/feature/train_imagenet_feature.npy", "/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature.npy")
    tsne_visualization("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature_all.npy", "/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy")
    # pca_reduction(np.load(src.config.IMG_FEATURE_FILE), src.config.IMG_FEATURE_DIM, save_path=src.config.REDUCTED_IMG_FEATURE_FILE)
