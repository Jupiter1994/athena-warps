#!/bin/bash
#SBATCH --account=b1094 ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=ciera-std ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --time=2:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes=1 ## how many computers/nodes do you need (no default)
#SBATCH --ntasks-per-node=32 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem=2G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=disk-sph1 ## When you run squeue -u
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jupiterding2029@u.northwestern.edu

# load in modules
module load python-anaconda3
module load intel/2024.0
module load mpi/intel-mpi-5.1.3.258
module load hdf5/1.10.7-openmpi-intel-2021.4.0

# configure problem
python ../../configure.py --prob=disk_constant_alpha --coord=spherical_polar --eos=isothermal -hdf5 -h5double -mpi --hdf5_path /hpc/software/spack_v20d1/spack/opt/spack/linux-rhel7-x86_64/intel-2021.4.0/hdf5-1.10.7-exmishiaff3ugqbeeimkqvnwrir76sxh -omp --cflag=-qno-openmp-simd
make clean
make -j 4

# run the sim
mpiexec -n ${SLURM_NTASKS} ../../bin/athena -i athinput.disk_sph
