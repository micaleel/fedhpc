# fedhpc

A friendly Federated Learning framework designed to work on HPC clusters.

## Install

```bash
make install
```

You can remove the conda environment using `make uninstall`.

## Running

```bash
mpirun -n <num_nodes> ./federated.py 
```