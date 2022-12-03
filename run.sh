#!/bin/bash
#SBATCH --nodes=20		    # number of nodes (-N)
#SBATCH --time=00:10:00		# Wall time
#SBATCH -A EuroCC-AF-6      # Project Account	
#SBATCH -J ucdfedhpc
#SBATCH -e error-%j.txt
#SBATCH -o output.txt       # Write stdout+stderr to file
#SBATCH -p ProdQ

# Mail me on job start & end
#SBATCH --mail-user=khalil.muhammad@ucd.ie
#SBATCH --mail-type=BEGIN,END

module load conda/2 openmpi/gcc/4.0.2rc2-ucx-cuda

source activate fedhpc

srun python ./federated.py