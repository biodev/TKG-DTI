#!/bin/bash

##############################################################
##############################################################
##############################################################

# configs

## data args
# Assumes that the data has been processed and is present at the ROOT folder. If not, see notebooks/tkg/ for data processing.
ROOT=../../data/tkg/
OUT=../../output/tkge/
DATA=$ROOT/processed/
TARGET_RELATION="drug,targets,gene"
K=10

## complex args 

OPTIM=adam
WD=1e-6
CHANNELS=512
BATCH_SIZE=5000
N_EPOCHS=100
NUM_WORKERS=10
LR=1e-3
DROPOUT=0.
LOG_EVERY=1
PATIENCE=100
TARGET_METRIC=mrr

## GNN args 

# hparam results: 

#lr	    wd	            channels	layers	batch_size	heads	norm	conv	nonlin	dropout	edge_dim	residual	val_MRR	val_avg_AUC
#0.001	1.000000e-07	12	        4	    5	        2	    layer	gat	    mish	0.1	    6	        True	    0.174691	0.984053

WD2=1e-7
CHANNELS2=12
LAYERS2=4
BATCH_SIZE2=5
N_EPOCHS2=100
NUM_WORKERS2=10
LR2=1e-3
DROPOUT2=0.1
LOG_EVERY2=1
NONLIN2=mish
HEADS2=2
EDGE_DIM2=6
CONV2=gat
PATIENCE2=15
##############################################################
##############################################################
##############################################################

echo ""
echo ""

for ((i=0; i<K; i++)); 
do 
    echo 'training GNN...'
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
    --conv $CONV2 \
    --residual

    echo 'training complex2...'
    python ../train_complex2.py --data $DATA/FOLD_$i/ \
    --out $OUT/COMPLEX2/FOLD_$i/ \
    --optim $OPTIM \
    --wd $WD \
    --channels $CHANNELS \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --dropout $DROPOUT \
    --log_every $LOG_EVERY \
    --patience $PATIENCE \
    --target_relation $TARGET_RELATION \
    --target_metric $TARGET_METRIC


done