#!/bin/bash -l

# Queue the job (time format = hh:mm:ss)
#SBATCH --job-name="geo_LWE_prediction_verification"
#SBATCH --qos=default
#SBATCH --time=18:00:00
#SBATCH --mail-type=ALL

# Set the specifications
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2gb
#SBATCH --array=0-35

# Set stdout/stderr
#SBATCH --output="output/prediction_verification/%j.out"
#SBATCH --error="output/prediction_verification/%j.err"

## No more SBATCH commands after this point ##

# Load slurm modules (needed software)
# Source scripts for loading modules in bash
. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh

module add Python3/3.10.6
module add sage

# Define and create unique scratch directory for this job
SCRATCH_DIRECTORY=/scratch0/${USER}/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}/
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Copy code to the scratch directory
cp -r ${SLURM_SUBMIT_DIR}/geometricLWE ${SCRATCH_DIRECTORY}

# Run code
cd geometricLWE/CKKS_experiments
sage CKKS_prediction_verification.sage 200 10 ${SLURM_ARRAY_TASK_ID}

# Remove code files
rm -rf ${SCRATCH_DIRECTORY}

# Finish the script
exit 0
