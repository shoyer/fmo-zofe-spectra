#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -m n
#
. $HOME/.bash_profile
python $1 --job-id $JOB_ID --job-name $JOB_NAME --task-id $SGE_TASK_ID
