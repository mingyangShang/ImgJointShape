import numpy as np
import math
import argparse
import subprocess
import tempfile
from scipy.spatial.distance import *
import os

parser = argparse.ArgumentParser(description="shape2image and image2shape evaluation on exact-match-chair dataset.")
parser.add_argument('-fs', '--shape_embedding_file', help='Shape embedding txt file (#model * #embedding-dim).',
                    required=False)
parser.add_argument('-fi', '--img_embedding_file', help='Image embedding txt file')
parser.add_argument('-d', '--model_deploy_file', help='Caffe model deploy file (batch size = 50).', required=False)
parser.add_argument('-p', '--model_param_file', help='Caffe model parameter file.', required=False)
parser.add_argument('-n1', '--nb_image2shape', help='Number of nearest shapes', required=True)
parser.add_argument('-n2', '--nb_shape2image', help='Number of nearest images', required=True)
parser.add_argument('--data', '--exact_match_dataset', help='path of exact match dataset', required=True)
parser.add_argument('--result_dir', help='path of result dir', required=True)
parser.add_argument('--distance_matrix_txt_file', help='Distance matrix (#image * #model) txt file (default=None)',
                    default=None, required=False)
parser.add_argument('--clutter_only', help='Test on clutter image only.', action='store_true')
parser.add_argument('--clean_only', help='Test on clean image only.', action='store_true')
parser.add_argument('--feat_dim', help='Embedding feat dim (default=100)', default=100, required=False)

args = parser.parse_args()

EXACTMATCH_DATASET = args.data
RESULTDIR = args.result_dir

# if 10models, D = D[k, eact_match_modelIds]
def img2shape(img_fcs, shape_fcs, pair_img_model, tag="all"):
    D = cdist(img_fcs, shape_fcs)
    image_N = D.shape[0]
    print image_N
    # Img2Shape
    image2shape_retrieval_ranking = []
    for k in range(image_N):
        distances = D[k, :]  # [float(distance) for distance in line.strip().split()]
        ranking = range(len(distances))
        ranking.sort(key=lambda rank: distances[rank])
        print 'image %d \t retrieval: %d' % (k, ranking.index(pair_img_model[k]) + 1)
        image2shape_retrieval_ranking.append(ranking.index(pair_img_model[k]) + 1)
    image2shape_topK_accuracies = []
    for topK in range(250):
        n = sum([r <= topK + 1 for r in image2shape_retrieval_ranking])
        image2shape_topK_accuracies.append(n / float(image_N))
    np.savetxt(os.path.join(RESULTDIR, 'image2shape_topK_accuracy_%s.txt'%tag), image2shape_topK_accuracies, fmt='%.4f')

def test_img2shape():
    fcs_img = np.zeros([10, 128])
    for i in range(fcs_img.shape[0]):
        fcs_img[i] = fcs_img[i] + i
    fcs_shape = fcs_img
    print fcs_img.shape, fcs_shape.shape
    fake_pair_img2shape = np.arange(0, fcs_img.shape[0])
    # retrieval result will be all 1
    img2shape(fcs_img, fcs_shape, fake_pair_img2shape, "fake_shapenet")

if __name__ == '__main__':
    fcs_img, fcs_shape = np.load(args.img_embedding_file), np.load(args.shape_embedding_file)
    # fcs_img = np.zeros([300, 128])
    # for i in range(300):
    #     fcs_img[i] = fcs_img[i] + i
    # fcs_shape = fcs_img
    print fcs_img.shape, fcs_shape.shape
    fake_pair_img2shape = np.arange(0, fcs_img.shape[0])
    img2shape(fcs_img, fcs_shape, fake_pair_img2shape, "fake_shapenet")
