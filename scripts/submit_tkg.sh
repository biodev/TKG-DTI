
logs=~/slurm_logs/tkg/
mkdir -p $logs

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=tkg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=$logs/log.%j.out
#SBATCH --error=$logs/log.%j.err

source ~/.zshrc
cd /path/to/TKG-DTI/workflow/aml-tkg/  # UPDATE: set to your TKG-DTI path
conda activate tkgdti
snakemake --unlock
snakemake -j 1

EOF