PYTHON=/home/users/nus/e1083772/.localpython/bin/python3.9
WDIR=/home/users/nus/e1083772/cancer-survival-ml

clean:
	find .pbs -type f -mmin +1 -delete

train:
	script=$(WDIR)/modules_vae/main.py; \
	endpoints=("pfs" "os"); \
	for endpoint in $${endpoints[@]}; do \
		qsub -p 0 -j oe -o $(WDIR)/.pbs -q normal -J 0-49 -l select=1:ncpus=2:mem=1024mb -l walltime=02:00:01 -- $(PYTHON) $$script --endpoint $$endpoint; \
	done

eval:
	qsub -p 0 -j oe -o $(WDIR)/.pbs -q normal -l select=1:ncpus=2:mem=1024mb -l walltime=02:00:01 -- $(PYTHON) $(WDIR)/modules_vae/eval.py
	