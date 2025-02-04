#!/bin/zsh

DATA=../data/tkg/processed/FOLD_0/
OUT=../output/hparam_search/gnn/FOLD_0/
N=25 # number of jobs for h param search to submit; "budget"
TIME=12:00:00
SEARCHSPACE=hp
OUT2=$OUT/logs/
PATIENCE=10

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=tune$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=$TIME
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

cd .. 
source ~/.zshrc
conda activate tkgdti
python gnn_hparam_tuning.py --data $DATA --out $OUT --n_runs $N --searchspace $SEARCHSPACE --patience $PATIENCE

EOF