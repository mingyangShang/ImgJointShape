shape_embedding_file='/home1/shangmingyang/data/ImgJoint3D/feature/eval_shape_feature.npy'
img_embedding_file='/home1/shangmingyang/data/ImgJoint3D/feature/eval_img_joint_feature.npy'
#img_embedding_file='/home/shangmingyang/projects/ImgJoint3D/src/experiments/fake_image_embedding.npy'
# figure 10 our embedding
BASEDIR="$(dirname $(readlink -f $0))"
RESULTDIR=$BASEDIR/results
EXACTMATCH_DATASET=$BASEDIR/ExactMatchChairsDataset
DISTANCE_MATRIX_DIR=$BASEDIR/compute_distance_matrices
if [ ! -d $RESULTDIR ]
then
    mkdir $RESULTDIR
fi
# figure 10 our embedding
python $BASEDIR/img2shape.py -m $EXACTMATCH_DATASET/exact_match_chairs_img_modelIds_0to6776.txt -fi $img_embedding_file -fs $shape_embedding_file   -n1 250 -n2 250 --result_id $RESULTDIR/test_clutter --clutter_only
# table 2 our embedding
#python $BASEDIR/img2shape.py -m $EXACTMATCH_DATASET/exact_match_chairs_img_modelIds_0to6776.txt -fi $img_embedding_file -fs $shape_embedding_file -n1 250 -n2 250 --result_id $RESULTDIR/sammon100_tmp