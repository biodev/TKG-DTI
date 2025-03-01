#!/bin/bash

##############################################################
##############################################################
##############################################################

# configs

## data args
# Assumes that the data has been processed and is present at the ROOT folder. If not, see notebooks/tkg/ for data processing.
ROOT=../../data/tkg/processed/FOLD_0/
OUT=../../output/hparam_tuning/tkg/gnn/
LOGDIR=$OUT/logs/
DATA=$ROOT/processed/
TARGET_RELATION="drug,targets,gene"

N=2 # compute budget (number of runs)

TIME=04:00:00
MEM=24G
CPUS=16

NUM_WORKERS=$CPUS
N_EPOCHS=100
PATIENCE=10

## complex args 


##############################################################
##############################################################
##############################################################



echo ""
echo ""

# loop through idxs 
for idx in $(seq 0 $N_RELATIONS); do
    for ((j=0; j<N; j++)); do
        echo 'submitting job for relation ablation...idx='$idx', fold='$i', run='$j

        sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=gnn$idx
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

python ../gnn_hparam_run.py --data $DATA --out $OUT --n_epochs N_EPOCHS --num_workers NUM_WORKERS --patience PATIENCE

###########################################################


EOF

done
done