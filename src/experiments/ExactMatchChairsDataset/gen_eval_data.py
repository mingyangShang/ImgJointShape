import numpy as np
import csv
import os
import argparse

import src.config as config

class2label = {
    "airplane": 0, "car": 1, "chair": 2
}

def gen_eval_img_data():
    print("Generate eval image data")
    eval_img_features = np.load(config.IMG_TEST_FEATURE_FILE)
    eval_img_labels = np.zeros([eval_img_features.shape[0]], dtype=int) + class2label["chair"]
    np.save(config.IMG_TEST_LABEL_FILE, eval_img_labels)
    print("Eval image labels saved to %s" % config.IMG_TEST_LABEL_FILE)

def gen_eval_shape_data():
    gen_eval_shape_index()
    print("Index of eval shape in v2 saved to %s"%os.path.abspath('eval_shape_in_v2_index.txt'))
    eval_index = np.loadtxt("eval_shape_in_v2_index.txt").astype(int)
    all_shape_features = np.load(config.SHAPENET_ALL_FEATURE_FILE)
    eval_shape_features = all_shape_features[eval_index, :]
    np.save(config.SHAPE_TEST_FEATURE_FILE, eval_shape_features)
    print("Eval shape features saved to %s" % config.SHAPE_TEST_FEATURE_FILE)
    eval_shape_labels = np.zeros([eval_shape_features.shape[0]], dtype=int) + class2label["chair"]
    np.save(config.SHAPE_TEST_LABEL_FILE, eval_shape_labels)
    print("Eval shape labels saved to %s" % config.SHAPE_TEST_LABEL_FILE)

def gen_eval_shape_index():
    v1_models = [x.rstrip() for x in open(config.EXPERIMENTS_EVAL_CHAIR_FILE, 'r')]
    v2_models = {}
    index = 0
    with open(config.SHAPENET_METADATA_FILE, 'r') as f:
        f.readline()
        r = csv.reader(f)
        for row in r:
            v2_models[row[3]] = index
            index += 1

    v1_index = [v2_models[v1_m]  for v1_m in v1_models]
    np.savetxt("eval_shape_in_v2_index.txt", np.array(v1_index))

def fake_eval_data():
    train_shape_fcs, train_img_fcs = np.load(config.SHAPE_TRAIN_FEATURE_FILE), np.load(config.IMG_TRAIN_FEATURE_FILE)
    fake_eval_shape_fcs, fake_eval_img_fcs = train_shape_fcs[:300], train_img_fcs[:300]
    np.save(config.SHAPE_TEST_FEATURE_FILE, fake_eval_shape_fcs)
    np.save(config.IMG_TEST_FEATURE_FILE, fake_eval_img_fcs)
    train_shape_labels, train_img_labels = np.load(config.SHAPE_TRAIN_LABEL_FILE), np.load(config.IMG_TRAIN_LABEL_FILE)
    np.save(config.SHAPE_TEST_LABEL_FILE, train_shape_labels[:300])
    np.save(config.IMG_TEST_LABEL_FILE, train_shape_labels[:300])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-eval-image", action='store_true', help='whether generate eval image data')
    parser.add_argument('--gen-eval-shape', action='store_true', help='whether generate eval shape data')
    parser.add_argument('--fake-eval',action='store_true', help='whether fake eval data')
    args = parser.parse_args()
    if args.gen_eval_image:
        gen_eval_img_data()
    if args.gen_eval_shape:
        gen_eval_shape_data()
    if args.fake_eval:
        fake_eval_data()
