import os
root_dir = '/home1/shangmingyang/projects/ImgJointShape'
# Features and labels extracted for shapenet
SHAPENET_TEST_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shapenet55_nocolor_test_features_512.npy')
SHAPENET_TRAIN_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shapenet55_nocolor_train_features_512.npy')
SHAPENET_VAL_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shapenet55_nocolor_val_features_512.npy')
SHAPENET_ALL_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shapenet55_nocolor_features_all.npy')
SHAPENET_TEST_LABEL_FILE = os.path.join(root_dir, 'data/shape/shapenet55_v1_test_labels.npy')
SHAPENET_TRAIN_LABEL_FILE = os.path.join(root_dir, 'data/shape/shapenet55_v1_train_labels.npy')
SHAPENET_VAL_LABEL_FILE = os.path.join(root_dir, 'data/shape/shapenet55_v1_val_labels.npy')
SHAPENET_ALL_LABEL_FILE = os.path.join(root_dir, 'data/shape/shapenet55_v1_all_labels.npy')

# Shapenet meta data
SHAPENET_METADATA_FILE = os.path.join(root_dir, 'data/shape/cleaned_all.csv')

# Features and labels extracted for imagenet
IMAGENET_DIR = os.path.join(root_dir, 'data/image/imagenet')
IMAGENET_ALL_FEATURE_FILE = os.path.join(root_dir, 'data/image/image_all_features_vgg.npy')
IMG_TRAIN_FEATURE_FILE = os.path.join(root_dir, 'data/image/image_train_features_vgg.npy')
IMG_TRAIN_LABEL_FILE = os.path.join(root_dir, 'data/image/image_train_labels.npy')

# Model train data
SHAPE_TRAIN_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shape_train_features_512.npy')
SHAPE_TRAIN_LABEL_FILE = os.path.join(root_dir, 'data/shape/shape_train_labels.npy')


IMG_FEATURE_FILE = '/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature.npy'
# IMG_FEATURE_FILE = '/home1/shangmingyang/data/ImgJoint3D/feature/eval_img_feature.npy'
IMG_FEATURE_DIM = 128
REDUCTED_IMG_FEATURE_FILE = '/home1/shangmingyang/data/ImgJoint3D/feature/train_imagenet_feature.npy'
# REDUCTED_IMG_FEATURE_FILE = '/home1/shangmingyang/data/ImgJoint3D/feature/eval_img_feature_reducted.npy'
SHAPE_FEATURE_FILE = '/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature.npy'

SAVE_MODEL_FILE = '/home1/shangmingyang/data/ImgJoint3D/model/model.pkl'

TEST_IMG_FEATURE = '/home1/shangmingyang/data/ImgJoint3D/feature/eval_img_feature.npy'
TEST_SHAPE_FEATURE = '/home1/shangmingyang/data/ImgJoint3D/feature/eval_shape_feature.npy'
TEST_IMG_JOINT_FEATURE = '/home1/shangmingyang/data/ImgJoint3D/result/eval_img_joint_feature.npy'
