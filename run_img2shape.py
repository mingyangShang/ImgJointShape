import os
import src.config as config

# generate joint feature
os.system('python -m src.ACMR.eval_adv_crossmodal_triplet_wiki')

if not os.path.exists(config.EXPERIMENTS_EVAL_RESULT_DIR):
    os.makedirs(config.EXPERIMENTS_EVAL_RESULT_DIR)

cmd = 'python -m src.experiments.img2shape -fi %s -fs %s -n1 250 -n2 250 --data %s --result_dir %s --clutter_only' \
    %(config.EXPERIMENTS_EVAL_IMG_FEATURE_FILE, config.EXPERIMENTS_EVAL_SHAPE_FEATURE_FILE, config.EXPERIMENTS_EXACT_MATCH_DATASET_DIR, config.EXPERIMENTS_EVAL_RESULT_DIR)
os.system(cmd)