#!/bin/bash
#SBATCH -c 1                               
#SBATCH -t 0-12:00
#SBATCH -p short 
#SBATCH --mem=4096M                       
#SBATCH -o rebind_bs_%A_%a.out              	   
#SBATCH -e rebind_bs_%A_%a.err                  

source activate rebind
python3 tx_fitting.py ./num_expt/txFit001/txFit001.yaml ./num_expt/txFit001/txFit001_res_0422_bs.yaml --tol=5 --default-init --bootstrap
