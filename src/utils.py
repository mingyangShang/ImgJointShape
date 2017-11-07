import numpy as np
import os
import matplotlib.pyplot as plt

def generate_class_center(features, labels):
    unique_labels = np.unique(labels)
    label_center_map = {}
    for l in unique_labels:
        label_center_map[l] = np.mean(features[np.where(labels==l)], axis=0)
    return label_center_map

def test_generate_class_center():
    features = np.array([[1,2,3,4],[5,5,5,5], [6,6,6,6]])
    labels = np.array([0,1,0])
    label_center_map = generate_class_center(features, labels)
    print label_center_map

def generate_shape_class_center(feature_file, label_file, save_dir=""):
    label_center_map = generate_class_center(np.load(feature_file), np.load(label_file))
    labels, features = label_center_map.keys(), label_center_map.values()
    np.save(os.path.join(save_dir, "shape_center_features.npy"), np.array(features))
    np.save(os.path.join(save_dir, "shape_center_labels.npy"), np.array(labels))

def reduct_class_center():
    from reduction import pca_reduction, tsne_reduction
    shape_center_features = np.load('/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_features.npy')
    pca_reduction(shape_center_features, 2, '/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_features_2dim_pca.npy')
    tsne_reduction(shape_center_features, 2, '/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_features_2dim_tsne.npy')

def visualize_class_center(pca_feature_file, tsne_feature_file, label_file, fig_index=0):
    from visualization import feature_reduction_vis
    pca_reducted_features = np.load(pca_feature_file)
    tsne_reducted_features = np.load(tsne_feature_file)
    shape_labels = np.load(label_file)
    import matplotlib.pyplot as plt
    # feature_reduction_vis(pca_reducted_features, shape_labels, tag="pca", fig_index=0)
    feature_reduction_vis(tsne_reducted_features, shape_labels, tag="tsne", fig_index=fig_index)



if __name__ == '__main__':
    # generate_shape_class_center("/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy", \
    #                             "/home1/shangmingyang/data/ImgJoint3D/feature/all_shapenet55_labels.npy", \
    #                             "/home1/shangmingyang/data/ImgJoint3D/feature/")
    visualize_class_center("/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_features_2dim_pca.npy", \
                            "/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_features_2dim_tsne.npy", \
                            "/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_labels.npy", fig_index=0)
    visualize_class_center("/home1/shangmingyang/data/ImgJoint3D/feature/imagenet_class_center_feature_pca_2dim.npy", \
                           "/home1/shangmingyang/data/ImgJoint3D/feature/imagenet_class_center_feature_tsne_2dim.npy", \
                           "/home1/shangmingyang/data/ImgJoint3D/feature/shape_center_labels.npy", fig_index=1)
    plt.show()