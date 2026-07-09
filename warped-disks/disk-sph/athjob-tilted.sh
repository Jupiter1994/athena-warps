#!/bin/bash
#SBATCH --account=b1094 ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=ciera-std ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --time=2:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes=8 ## how many computers/nodes do you need (no default)
#SBATCH --ntasks-per-node=13 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem=3G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=disk-tilted27 ## When you run squeue -u 
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jupiterding2029@u.northwestern.edu

module load intel/2024.0
module load mpi/intel-mpi-5.1.3.258
module load hdf5/1.10.7-openmpi-intel-2021.4.0

mpiexec -n ${SLURM_NTASKS} ../../bin/athena -i athinput.disk_tilted
#mpiexec -n 416 ./athena -r disk.00025.rst -i athinput.disk_sph
