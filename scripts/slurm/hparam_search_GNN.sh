#!/bin/zsh

DATA=../data/heteroa/processed/FOLD_0/
OUT=../output/gnn/hparam_search/test/
N=1 # number of jobs for h param search to submit; "budget"

# parameter search grid
lr_list=("0.001" "0.0001")
do_list=("0")
c_list=("32" "64")
lay_list=("4" "6", "8")
conv_list=("gat")

mkdir $OUT

OUT2=$OUT/slurm_logs__GNN/
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
        do=$(echo "${do_list[@]}" | tr ' ' '\n' | shuf -n 1)
        c=$(echo "${c_list[@]}" | tr ' ' '\n' | shuf -n 1)
        lay=$(echo "${lay_list[@]}" | tr ' ' '\n' | shuf -n 1)
        conv=$(echo "${conv_list[@]}" | tr ' ' '\n' | shuf -n 1)

        jobid=$((jobid+1))

        echo "submitting job: GSNN (lr=$lr, do=$do, c=$c, lay=$lay, ase=$ase)"

        # SUBMIT SBATCH JOB 

        sbatch <<EOF

#!/bin/bash
#SBATCH --job-name=tkgdti$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

source ~/.zshrc
conda activate tkgdti
cd /home/exacloud/gscratch/NGSdev/evans/TKG-DTI/scripts/
python train_gnn.py --data $DATA --out $OUT --channels $c --conv $conv --num_workers 10 --layers $lay --dropout $do --lr $lr --epochs 100 --batch_size 1 --patience 5 --edge_dim 12 --residual 

EOF
done
done
done
done