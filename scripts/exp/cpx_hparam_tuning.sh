#!/bin/bash

##############################################################
##############################################################
##############################################################

# configs

## data args
DATA=/home/exacloud/gscratch/mcweeney_lab/evans/TKG-DTI/data/tkg/processed/FOLD_0/
OUT=/home/exacloud/gscratch/mcweeney_lab/evans/TKG-DTI/output/hparam_tuning/tkg/FOLD_0/complex2/
LOGDIR=$OUT/logs/

N=1 # compute budget (number of runs)

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

# loop through idxs 
for ((jj=0; jj<N; jj++)); do
    echo 'submitting hparam tuning run...i='$jj

    sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=cpx$jj
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
python ../cpx_hparam_run.py --data $DATA --out $OUT --n_epochs $N_EPOCHS --num_workers $NUM_WORKERS --patience $PATIENCE

###########################################################
EOF

done