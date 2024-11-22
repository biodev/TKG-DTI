#/bin/zsh

DATA=../data/heteroa/processed/FOLD_0/
OUT=../output/hparam_search/FOLD_0/gnn/
N=3 # number of jobs for h param search to submit; "budget"
NEPOCHS=3
TIME=01:00:00

# parameter search grid
lr_list=("0.01" "0.001" "0.0001")
do_list=("0" "0.1")
c_list=("32" "64")
lay_list=("5" "6" "7")
conv_list=("gat" "transformer")
edge_dim_list=("4" "8" "12")
mkdir $OUT

OUT2=../$OUT/slurm_logs__GNN/
if [ -d "$OUT2" ]; then
        echo "slurm output log dir exists. Erasing contents..."
        rm -r "$OUT2"/*
else
        echo "slurm output log dir does not exist. Creating..."
        mkdir "$OUT2"
fi

jobid=0
# LIMITED HYPER-PARAMETER SEARCH ; randomly sample from possible params 
for ((i=1; i<=N; i++)); do
        lr=$(echo "${lr_list[@]}" | tr ' ' '\n' | shuf -n 1)
        d=$(echo "${do_list[@]}" | tr ' ' '\n' | shuf -n 1)
        c=$(echo "${c_list[@]}" | tr ' ' '\n' | shuf -n 1)
        lay=$(echo "${lay_list[@]}" | tr ' ' '\n' | shuf -n 1)
        conv=$(echo "${conv_list[@]}" | tr ' ' '\n' | shuf -n 1)
	edim=$(echo "${edge_dim_list[@]}" | tr ' ' '\n' | shuf -n 1)
        jobid=$((jobid+1))

        echo "submitting job: KGDTI - GNN (lr=$lr, do=$d, c=$c, lay=$lay, conv=$conv, edge dim=$edim, jobid=$jobid)"

# SUBMIT SBATCH JOB 

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=gnn$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=$TIME
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

source ~/.zshrc
conda activate tkgdti
cd /home/exacloud/gscratch/NGSdev/evans/TKG-DTI/scripts/
python train_gnn.py --data $DATA --out $OUT --channels $c --conv $conv --num_workers 10 --layers $lay --dropout $d --lr $lr --n_epochs $NEPOCHS --batch_size 1 --patience 5 --edge_dim $edim --residual 

EOF
done
