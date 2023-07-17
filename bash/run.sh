#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -o out/run.out
#BSUB -e out/run.err
#BSUB -J run
cd ..
python run.py --device cuda
