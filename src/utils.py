import numpy as np


def generate_class_center(features, labels):
    unique_labels = np.unique(labels)
    label_center_map = {}
    for l in unique_labels:
        label_center_map[l] = np.mean(features[np.where(labels==l)], axis=0)
    return label_center_map


if __name__ == '__main__':
    generate_class_center("/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy", "")