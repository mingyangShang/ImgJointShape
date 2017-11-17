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
IMG_TRAIN_FEATURE_FILE = os.path.join(root_dir, 'data/image/shapenet_img_first/image_shapenet_first_train_features_vgg.npy')
# IMG_TRAIN_FEATURE_FILE = os.path.join(root_dir, 'data/image/shapenet_img_first/image_shapenet_first2_train_features_vgg.npy')
IMG_TRAIN_LABEL_FILE = os.path.join(root_dir, 'data/image/image_train_labels.npy')

# Model train data
SHAPE_TRAIN_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shape_train_features_512.npy')
SHAPE_TRAIN_LABEL_FILE = os.path.join(root_dir, 'data/shape/shape_train_labels.npy')

# Eval data
# IMG_EVAL_FEATURE_FILE = os.path.join(root_dir, 'data/image/image_eval_features_vgg.npy')
# IMG_EVAL_LABEL_FILE = os.path.join(root_dir, 'data/image/image_eval_labels.npy')
# SHAPE_EVAL_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shape_eval_features_512.npy')
# SHAPE_EVAL_LABEL_FILE = os.path.join(root_dir, 'data/shape/shape_eval_labels.npy')
IMG_EVAL_FEATURE_FILE = os.path.join(root_dir, 'data/image/image_eval_features_vgg_first.npy')
IMG_EVAL_LABEL_FILE = os.path.join(root_dir, 'data/image/image_eval_labels_first.npy')
SHAPE_EVAL_FEATURE_FILE = os.path.join(root_dir, 'data/shape/shape_eval_features_512_first.npy')
SHAPE_EVAL_LABEL_FILE = os.path.join(root_dir, 'data/shape/shape_eval_labels_first.npy')

EXPERIMENTS_EVAL_IMG_FEATURE_FILE = os.path.join(root_dir, 'data/result/image_eval_joint_features.npy')
EXPERIMENTS_EVAL_SHAPE_FEATURE_FILE = os.path.join(root_dir, 'data/result/shape_eval_joint_features.npy')
EXPERIMENTS_EXACT_MATCH_DATASET_DIR = os.path.join(root_dir, 'data/experiments/ExactMatchChairsDataset')
EXPERIMENTS_EVAL_CHAIR_FILE = os.path.join(EXPERIMENTS_EXACT_MATCH_DATASET_DIR, 'filelist_chair_6777.txt')
EXPERIMENTS_EVAL_CHAIR_IMG_MODELID_FILE = os.path.join(EXPERIMENTS_EXACT_MATCH_DATASET_DIR, 'exact_match_chairs_img_modelIds_0to6776.txt')
EXPERIMENTS_EVAL_RESULT_DIR = os.path.join(root_dir, 'data/experiments/results')


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
