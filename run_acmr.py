import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help='whether run in train mode')
    parser.add_argument("--eval", action='store_true', help='whether run in eval mode')
    args = parser.parse_args()
    if args.train:
        os.system('python -m src.ACMR.train_adv_crossmodal_triplet_wiki')
    if args.eval:
        os.system('python -m src.ACMR.eval_adv_crossmodal_triplet_wiki')

