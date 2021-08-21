#!/bin/bash
#SBATCH -n 10
#SBATCH -t 02:00:00
#SBATCH --mem=6G
#SBATCH -J amey_plot
#SBATCH -o plot_output.out
#SBATCH -e plot_error.err
module load python/3.9.0 mpi/openmpi_4.0.0_gcc hdf5/1.10.5_openmpi_4.0.0_gcc 
module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda create -n amey_conda
conda activate amey_conda
conda install numpy matplotlib numba h5py
python plot.py
conda deactivate