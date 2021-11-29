#!/bin/sh
#PBS -V
#PBS -q sugon10
#PBS -N tdm-polar
#PBS -l nodes=1:ppn=24
source /share/home/bjiangch/group-zyl/.bash_profile
# conda environment
conda_env=PyTorch-190
export OMP_NUM_THREADS=24
#path to save the code
path="/group/zyl/program/eann/program/"

#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
WORLD_SIZE=`expr $PBS_NUM_NODES \* $NPROC_PER_NODE`

MASTER=`/bin/hostname -s`

MPORT=`ss -tan | awk '{print $5}' | cut -d':' -f2 | \
        grep "[2-9][0-9]\{3,3\}" | sort | uniq | shuf -n 1`

#You will want to replace this
COMMAND="$path "
conda activate $conda_env 
cd $PBS_O_WORKDIR 
python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$PBS_NUM_NODES --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:1332 $COMMAND > out
#python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=1 --standalone $COMMAND > out

