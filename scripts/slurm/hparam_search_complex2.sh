#/bin/zsh

DATA=../data/heteroa/processed/FOLD_0/
OUT=../output/hparam_search/FOLD_0/complex2/
N=25 # number of jobs for h param search to submit; "budget"
NEPOCHS=500
TIME=04:00:00

# parameter search grid
lr_list=("0.01" "0.001" "0.0001")
c_list=("256" "512" "1024")
batch_list=("5000" "25000" "50000")
wd_list=("0" "1e-6" "1e-9")

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
        c=$(echo "${c_list[@]}" | tr ' ' '\n' | shuf -n 1)
        batch=$(echo "${batch_list[@]}" | tr ' ' '\n' | shuf -n 1)
        wd=$(echo "${wd_list[@]}" | tr ' ' '\n' | shuf -n 1)
        jobid=$((jobid+1))

        #echo "submitting job: KGDTI - GNN (lr=$lr, do=$d, c=$c, lay=$lay, conv=$conv, edge dim=$edim, jobid=$jobid)"
        echo "submitting job: KGDTI - GNN (lr=$lr, c=$c, batch=$batch, wd=$wd, jobid=$jobid)"

# SUBMIT SBATCH JOB 

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=cp2$jobid
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
python train_complex2.py --data $DATA --out $OUT --num_workers 10 --lr $lr --channels $c --batch_size $batch --wd $wd --n_epochs $NEPOCHS --patience 5 --log_every 1 --target_relations 5
EOF
done
