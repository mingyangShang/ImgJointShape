#!/usr/bin/env bash
source config
THEANO_FLAGS=floatX=float32,device=cpu python src/linear_ae_ff_stable.py --num-epoches 500 --update-times-g2d 10 --batch-size 256 --alt-loss --joint-dim 128 --train --recon-weight 0 > $reconWeight.log

#cut -d ' ' -f 1 data/$config/vocab-freq.$lang1 > data/$config/vocab.$lang1
#python3 scripts/translate.py data/$config/transformed-1.$lang1 data/$config/word2vec.$lang2 data/$config/vocab.$lang1 data/$config/result.1
