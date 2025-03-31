PYTHON=/home/users/nus/e1083772/.localpython/bin/python3
WDIR=/home/users/nus/e1083772/cancer-survival-ml
OPTS=-p 0 -j oe -o $(WDIR)/.pbs -q normal -l select=1:ncpus=4:mem=64G -l walltime=02:00:01
J ?= 0-1:2
EP ?= os # os pfs

SPLITDATA=${WDIR}/pipeline/1_split.py
PREPROCESS=$(WDIR)/pipeline/2_preprocess.py
FIT=$(WDIR)/pipeline/3_fit_vae.py
EVAL=$(WDIR)/pipeline/4_eval_vae.py

0:
	find $(WDIR)/.pbs -type f -delete

1:
	qsub ${OPTS} -J ${J} -- ${PYTHON} ${SPLITDATA}

2:
	qsub ${OPTS} -J ${J} -- ${PYTHON} ${PREPROCESS} --endpoint ${EP}

3:
	qsub $(OPTS) -J ${J} -- $(PYTHON) $(FIT) --endpoint ${EP}

4:
	qsub $(OPTS) -- $(PYTHON) $(EVAL) --endpoint ${EP}