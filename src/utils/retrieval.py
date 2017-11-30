import numpy as np
import os
from scipy.spatial.distance import *

def dis_img_shape(img_fcs, shape_fcs):
    return cdist(img_fcs, shape_fcs)
def dis_shape_img(shape_fcs, img_fcs):
    return cdist(shape_fcs, img_fcs)

def img2shape(img_fcs, shape_fcs, pair_img_model, top_k=50, tag="all", save_dir=""):
    D = cdist(img_fcs, shape_fcs)
    image_N = D.shape[0]
    image2shape_retrieval_ranking = []
    for k in range(image_N):
        distances = D[k, :]  # [float(distance) for distance in line.strip().split()]
        ranking = range(len(distances))
        ranking.sort(key=lambda rank: distances[rank])
        # print 'image %d \t retrieval: %d' % (k, ranking.index(pair_img_model[k]) + 1)
        image2shape_retrieval_ranking.append(ranking.index(pair_img_model[k]) + 1)
    image2shape_topK_accuracies = []
    for topK in range(top_k):
        n = sum([r <= topK + 1 for r in image2shape_retrieval_ranking])
        image2shape_topK_accuracies.append(n / float(image_N))
    if save_dir and len(save_dir) > 0:
        np.savetxt(os.path.join(save_dir, 'image2shape_topK_accuracy_%s.txt'%tag), image2shape_topK_accuracies, fmt='%.4f')
    return image2shape_topK_accuracies

def shape2img(shape_fcs, img_fcs, pair_model_img, tag="all", save_dir=""):
    pass

def test_img2shape():
    fcs_img = np.zeros([10, 128])
    for i in range(fcs_img.shape[0]):
        fcs_img[i] = fcs_img[i] + i
    fcs_shape = fcs_img
    fake_pair_img2shape = np.arange(0, fcs_img.shape[0])
    # retrieval result will be all 1
    print(img2shape(fcs_img, fcs_shape, fake_pair_img2shape, top_k=50, tag="fake_shapenet"))

if __name__ == '__main__':
    test_img2shape()

