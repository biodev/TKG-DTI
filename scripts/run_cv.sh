#!/bin/bash 

ROOT=$1
K=$2
CHANNELS=1024
OPTIM=adan 
BATCH_SIZE=10000
LR=1e-2
LOG_EVERY=1
PATIENCE=10


for ((i=0; i<K; i++)); 
do 
    python train_complex2.py --data ../data/tkg/processed/FOLD_$i/ --out ../output/tkg/FOLD_$i/ --channels $CHANNELS --optim $OPTIM --batch_size $BATCH_SIZE --lr $LR --log_every $LOG_EVERY --patience $PATIENCE
done