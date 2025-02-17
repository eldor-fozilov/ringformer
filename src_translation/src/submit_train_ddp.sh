#!/bin/bash 

source activate RF_13

cd $PBS_O_WORKDIR
DATE=`date +%y%m%d`
echo $PBS_JOBID

accelerate launch --multi_gpu --num_processes 2 --main_process_port 29697 main.py -d gpu -m train