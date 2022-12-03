ENVNAME=fedhpc

.ONESHELL:

install:
	@conda create -n $(ENVNAME) python=3.8 --yes
	@conda env update -n $(ENVNAME) -f environment.yml

uninstall:
	conda env remove -n $(ENVNAME)

run:
	mpirun -n $(n) python ./federated.py

srun:
	srun -A EuroCC-AF-6 -N 5 --time 00:10:00 mpirun -n 5 ./federated.py

submit:
	sbatch run.sh

push:
	rsync -avzP --delete --exclude '.git/' --exclude '/data/outputs' . khalil@kay.ichec.ie:/ichec/home/users/khalil/fedhpc

pushzip:
	git archive --output=./fedhpc.zip --format=zip HEAD
	scp fedhpc.zip khalil@kay.ichec.ie:/ichec/home/users/khalil
	rm fedhpc.zip

pull:
	scp -r khalil@kay.ichec.ie:/ichec/home/users/khalil/fedhpc/data/outputs .

history:
	scp khalil@kay.ichec.ie:/ichec/home/users/khalil/fedhpc/history-fed.png .

peek:
	squeue -A EuroCC-AF-6