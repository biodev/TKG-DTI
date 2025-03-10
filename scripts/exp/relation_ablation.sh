#!/bin/bash

##############################################################
##############################################################
##############################################################

# configs

## data args
# Assumes that the data has been processed and is present at the ROOT folder. If not, see notebooks/tkg/ for data processing.
ROOT=../../data/tkg/
OUT=../../output/ablation/tkge/
LOGDIR=$OUT/logs/
DATA=$ROOT/processed/
TARGET_RELATION="drug,targets,gene"

N_FULL=5
N=1
FOLD=0
N_RELATIONS=49  # 49 for testing use 2 


TIME=06:00:00
MEM=24G
CPUS=16

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
PATIENCE=10
TARGET_METRIC=mrr

## GNN args 

WD2=1e-7
CHANNELS2=16
LAYERS2=4
BATCH_SIZE2=5
N_EPOCHS2=100
NUM_WORKERS2=10
LR2=1e-2
DROPOUT2=0
LOG_EVERY2=1
NONLIN2=elu
HEADS2=2
EDGE_DIM2=4
CONV2=gat
PATIENCE2=10

##############################################################
##############################################################
##############################################################


for ((i=0; i<N_FULL; i++)); do
    
    sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=ablationFULL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=gpu:1
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --partition=gpu
#SBATCH --output=$LOGDIR/log.%j.out
#SBATCH --error=$LOGDIR/log.%j.err

source ~/.zshrc
conda activate tkgdti

python ../train_gnn.py --data $DATA/FOLD_$FOLD/ \
    --out $OUT/GNN/FOLD_$FOLD/ \
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

python ../train_complex2.py --data $DATA/FOLD_$FOLD/ \
    --out $OUT/COMPLEX2/FOLD_$FOLD/ \
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

EOF

done


echo ""
echo ""

# loop through idxs 
for idx in $(seq 0 $N_RELATIONS); do
    for ((j=0; j<N; j++)); do
        echo 'submitting job for relation ablation...idx='$idx', fold='$i', run='$j

        sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=ablation$idx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=gpu:1
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --partition=gpu
#SBATCH --output=$LOGDIR/log.%j.out
#SBATCH --error=$LOGDIR/log.%j.err

source ~/.zshrc
conda activate tkgdti

python ../train_gnn.py --data $DATA/FOLD_$FOLD/ \
    --out $OUT/GNN/FOLD_$FOLD/ \
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
    --remove_relation_idx $idx \
    --residual

python ../train_complex2.py --data $DATA/FOLD_$FOLD/ \
    --out $OUT/COMPLEX2/FOLD_$FOLD/ \
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
    --remove_relation_idx $idx

EOF

done
done