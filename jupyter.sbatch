#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=g40
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=--account
#SBATCH --gpus=neox
#SBATCH --cpus-per-gpu=12

cat /etc/hosts
export PATH="~/.local/bin:/admin/home-curt-tigges/.vscode-server/bin/ee2b180d582a7f601fa6ecfdad8d9fd269ab1884/bin/remote-cli:/opt/slurm/bin:/fsx/home-curt/miniconda3/bin:/fsx/home-curt/miniconda3/condabin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/slurm/bin:/opt/slurm/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
python3 -m pip install notebook
jupyter notebook --ip=0.0.0.0  
