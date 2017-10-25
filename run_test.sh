#!/usr/bin/env bash
batchSize=630 #315*2
THEANO_FLAGS=floatX=float32,device=cpu python src/linear_ae_ff_stable.py --batch-size $batchSize
