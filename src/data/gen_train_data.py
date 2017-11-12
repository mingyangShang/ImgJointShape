import numpy as np
import csv
import argparse
import glob
import os

import src.config as config

class2label = {
    "airplane": 0, "car": 1, "chair": 2
}

def gen_all_shape_data():
    print("Generate all shape data")
    # concantate all shape features and labels ordered by [test,train,val]
    shape_features_test, shape_features_train, shape_features_val = np.load(config.SHAPENET_TEST_FEATURE_FILE), np.load(config.SHAPENET_TRAIN_FEATURE_FILE), np.load(config.SHAPENET_VAL_FEATURE_FILE)
    shape_features_all = np.vstack([shape_features_test, shape_features_train, shape_features_val])
    np.save(config.SHAPENET_ALL_FEATURE_FILE, shape_features_all)
    print("All shape features saved to %s"%config.SHAPENET_ALL_FEATURE_FILE)
    shape_labels_test, shape_labels_train, shape_labels_val = np.load(config.SHAPENET_TEST_LABEL_FILE), np.load(config.SHAPENET_TRAIN_LABEL_FILE), np.load(config.SHAPENET_VAL_LABEL_FILE)
    shape_labels_all = np.concatenate([shape_labels_test, shape_labels_train, shape_labels_val])
    np.save(config.SHAPENET_ALL_LABEL_FILE, shape_labels_all)
    print("All shape labels saved to %s"%config.SHAPENET_ALL_LABEL_FILE)

def gen_train_shape_data():
    print("Generate train shape data")
    airplane_id, car_id, chair_id = "02691156", "02958343", "03001627"
    pre_class2label = {"airplane":11, "car":22, "chair":23}
    shape_index = {airplane_id:[], chair_id:[], car_id:[]}
    with open(config.SHAPENET_METADATA_FILE, 'r') as f:
        f.readline()
        r = csv.reader(f)
        i = 0
        for row in r:
            if row[1] in shape_index.keys():
                shape_index[row[1]].append(i)
            i += 1
    print("Airplane:%d, Car:%d, Chair:%d" %(len(shape_index[airplane_id]), len(shape_index[car_id]), len(shape_index[chair_id])))
    all_shape_features = np.load(config.SHAPENET_ALL_FEATURE_FILE)
    all_shape_labels = np.load(config.SHAPENET_ALL_LABEL_FILE)
    index = []
    for v in shape_index.values():
        index += v
    print shape_index.keys()
    np.save(config.SHAPE_TRAIN_FEATURE_FILE, all_shape_features[np.array(index), :])
    print("Train shape features saved to %s"%config.SHAPE_TRAIN_FEATURE_FILE)
    train_shape_labels = all_shape_labels[np.array(index)]
    # do label transform to get labels [0,2]
    for c, l in pre_class2label.items():
        train_shape_labels[train_shape_labels==l] = class2label[c]
    np.save(config.SHAPE_TRAIN_LABEL_FILE, train_shape_labels)
    print("Train shape labels saved to %s"%config.SHAPE_TRAIN_LABEL_FILE)

def gen_all_img_data():
    print("Generate all imagenet data")
    imagenet_classes_features_files = glob.glob(os.path.join(config.IMAGENET_DIR, "imagenet_*_feature.npy"))
    imagenet_classes_features = [np.load(f) for f in imagenet_classes_features_files]
    all_imagenet_features = np.vstack(imagenet_classes_features)
    np.save(config.IMAGENET_ALL_FEATURE_FILE, all_imagenet_features)
    print("All imagenet features saved to %s"%config.IMAGENET_ALL_FEATURE_FILE)

def gen_train_img_data():
    print("Generate train img data")
    train_img_features, train_img_labels = [], []
    for c in ["airplane", "car", "chair"]:
        features = np.load(os.path.join(config.IMAGENET_DIR, "imagenet_%s_feature.npy"%c))
        labels = np.zeros(features.shape[0], dtype=int) + class2label[c]
        train_img_features.append(features)
        train_img_labels.append(labels)
    np.save(config.IMG_TRAIN_FEATURE_FILE, np.vstack(train_img_features))
    print("Train img features saved to %s"%config.IMG_TRAIN_FEATURE_FILE)
    np.save(config.IMG_TRAIN_LABEL_FILE, np.concatenate(train_img_labels))
    print("Train img labels saved to %s"%config.IMG_TRAIN_LABEL_FILE)

def gen_train_data():
    airplane_id, chair_id, car_id = "02691156", "03001627", "02958343"
    shape_index = {airplane_id:[], chair_id:[], car_id:[]}
    chair_rows = []
    with open(config.SHAPENET_METADATA_FILE, 'r') as f:
        f.readline()
        r = csv.reader(f)
        i = 0
        for row in r:
            if row[1] in shape_index.keys():
                shape_index[row[1]].append(i)
            if row[1] == '03001627':
                chair_rows.append(row[3]+'\n')
            i += 1

    with open('../filelist_chair_6777.txt', 'r') as f:
        rows = set(f.readlines())
    print "chair_rows",chair_rows
    print rows
    result = list(rows.difference(set(chair_rows)))
    with open("chair_diff.txt", 'w') as f:
        f.writelines(result)


    print len(shape_index[airplane_id]), len(shape_index[chair_id]), len(shape_index[car_id])
    # all_shape_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy")
    # all_img_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature_all.npy")
    # index = []
    # for v in shape_index.values():
    #     index += v
    # np.savetxt("/home1/shangmingyang/data/ImgJoint3D/feature/train_index.txt", np.array(index))
    # np.savetxt("/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature.txt", all_shape_features[np.array(index), :])
    # np.save("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature", all_img_features[np.array(index), :, :])


def gen_eval_shape_index():
    v1_models = [x.rstrip() for x in open('../filelist_chair_6777.txt', 'r')]
    v2_models = {}
    index = 0
    with open("../cleaned_all.csv", 'r') as f:
        f.readline()
        r = csv.reader(f)
        for row in r:
            v2_models[row[3]] = index
            index += 1

    v1_index = [v2_models[v1_m]  for v1_m in v1_models]
    np.savetxt("../v1_in_v2_index.txt", np.array(v1_index))

def gen_eval_shape_feature():
    gen_eval_shape_index()
    eval_index = np.loadtxt("../v1_in_v2_index.txt").astype(int)
    # all_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy")
    all_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature_all.npy")
    eval_features = all_features[eval_index, :]
    np.save("/home1/shangmingyang/data/ImgJoint3D/feature/eval_shape_img_feature.npy", eval_features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-all-shape", action='store_true', help='whether generate all shape data')
    parser.add_argument('--gen-train-shape', action='store_true', help='whether generate train data')
    parser.add_argument("--gen-all-img", action='store_true', help='whether generate all image data')
    parser.add_argument('--gen-train-img', action='store_true', help='whether generate img data')
    args = parser.parse_args()
    if args.gen_all_shape:
        gen_all_shape_data()
    if args.gen_train_shape:
        gen_train_shape_data()
    if args.gen_all_img:
        gen_all_img_data()
    if args.gen_train_img:
        gen_train_img_data()