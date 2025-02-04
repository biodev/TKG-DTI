#!/bin/bash

##############################################################
##############################################################
##############################################################

# configs

## heteroA args 
ROOT=../../data/HeteroA/
URL=https://github.com/luoyunan/PyDTINet/raw/main/data.tar.gz
K=10
TRAIN_P=0.9
PP_THRESH=90
DD_THRESH=0.9
SEED=0
N_NEG_SAMPLES=10
DATA=$ROOT/processed/
OUT=../../output/HeteroA/
TARGET_RELATION="drug,drug->target->protein,protein"

## complex args 

OPTIM=adam
WD=0
CHANNELS=512
BATCH_SIZE=10000
N_EPOCHS=100
NUM_WORKERS=10
LR=1e-3
DROPOUT=0.
LOG_EVERY=10
PATIENCE=2
TARGET_METRIC=mrr

## GNN args 

WD2=0
CHANNELS2=12
LAYERS2=4
BATCH_SIZE2=5
N_EPOCHS2=100
NUM_WORKERS2=10
LR2=1e-3
DROPOUT2=0.25
LOG_EVERY2=2
PATIENCE2=3
NONLIN2=elu
HEADS2=1
EDGE_DIM2=4
NORM_MODE2=node
CONV2=gat
PATIENCE2=3
##############################################################
##############################################################
##############################################################

echo 'making heteroa...'
#python ../make_heteroa.py --root $ROOT \
#    --url $URL \
#    --k $K \
#    --train_p $TRAIN_P \
#    --pp_thresh $PP_THRESH \
#    --dd_thresh $DD_THRESH \
#    --seed $SEED \
#    --n_neg_samples $N_NEG_SAMPLES

echo ""
echo ""
echo 'training complex2...'

for ((i=0; i<K; i++)); 
do 
    #python ../train_complex2.py --data $DATA/FOLD_$i/ \
    #--out $OUT/COMPLEX2/FOLD_$i/ \
    #--optim $OPTIM \
    #--wd $WD \,.
    #--channels $CHANNELS \
    #--batch_size $BATCH_SIZE \
    #--n_epochs $N_EPOCHS \
    #--num_workers $NUM_WORKERS \
    #--lr $LR \
    #--dropout $DROPOUT \
    #--log_every $LOG_EVERY \
    #--patience $PATIENCE \
    #--target_relation $TARGET_RELATION \
    #--target_metric $TARGET_METRIC

    python ../train_gnn.py --data $DATA/FOLD_$i/ \
    --out $OUT/GNN/FOLD_$i/ \
    --wd $WD2 \
    --channels $CHANNELS2 \
    --layers $LAYERS2 \
    --batch_size $BATCH_SIZE2 \
    --n_epochs $N_EPOCHS2 \
    --num_workers $NUM_WORKERS2 \
    --lr $LR2 \
    --dropout $DROPOUT2 \
    --log_every $LOG_EVERY2 \
    --patience $PATIENCE2 \
    --nonlin $NONLIN2 \
    --heads $HEADS2 \
    --edge_dim $EDGE_DIM2 \
    --norm_mode $NORM_MODE2 \
    --conv $CONV2 \
    --residual \
    --heteroA

done