#!/bin/zsh

logs=~/slurm_logs/heteroa/
mkdir -p $logs

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=heteroa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=$logs/log.%j.out
#SBATCH --error=$logs/log.%j.err

source ~/.zshrc
cd /home/exacloud/gscratch/mcweeney_lab/evans/TKG-DTI/workflow/hetero-a/
conda activate tkgdti
snakemake --unlock
snakemake -j 1

EOF