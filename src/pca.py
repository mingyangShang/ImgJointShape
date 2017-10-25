import numpy as np
from sklearn.decomposition import PCA

import src.config


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

if __name__ == "__main__":
    pca_reduction(np.load(src.config.IMG_FEATURE_FILE), src.config.IMG_FEATURE_DIM, save_path=src.config.REDUCTED_IMG_FEATURE_FILE)
