PYTHON=/home/users/nus/e1083772/.localpython/bin/python3.9
WDIR=/home/users/nus/e1083772/cancer-survival-ml
TRAINSCRIPT=$(WDIR)/modules_vae/main.py
EVALSCRIPT=$(WDIR)/modules_vae/eval.py
PBSOPTS=-p 0 -j oe -o $(WDIR)/.pbs -q normal -l select=1:ncpus=2:mem=1024mb -l walltime=02:00:01

all: pfs os clean

one:
	qsub $(PBSOPTS) -J 0-1:2 -- $(PYTHON) $(TRAINSCRIPT) --endpoint "pfs"	

pfs:
	qsub $(PBSOPTS) -J 0-49 -- $(PYTHON) $(TRAINSCRIPT) --endpoint "pfs"

os:
	qsub $(PBSOPTS) -J 0-49 -- $(PYTHON) $(TRAINSCRIPT) --endpoint "os"

clean:
	find .pbs -type f -mmin +1 -delete

eval:
	qsub $(PBSOPTS) -- $(PYTHON) $(EVALSCRIPT)
	