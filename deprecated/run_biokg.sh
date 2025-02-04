#!/bin/bash

##############################################################
##############################################################
##############################################################

# configs

## biokg args 

ROOT=../../data/biokg/

## complex args 

DATA=$ROOT/processed/
OUT=../output/biokg/
OPTIM=adam
WD=0
CHANNELS=1024
BATCH_SIZE=10000
N_EPOCHS=100
NUM_WORKERS=10
LR=1e-3
DROPOUT=0.
LOG_EVERY=5
PATIENCE=3
TARGET_RELATION=drug,drug-protein,protein
TARGET_METRIC=mrr

##############################################################
##############################################################
##############################################################

echo 'making biokg...'
#python -W ignore::FutureWarning ../make_biokg.py --root $ROOT

echo ""
echo ""
echo 'training complex2...'
python -W ignore::FutureWarning ../train_complex2.py --data $DATA \
    --out $OUT \
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
    --target_metric $TARGET_METRIC \


