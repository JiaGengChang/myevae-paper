WDIR=/home/users/nus/e1083772/cancer-survival-ml
NCPUS ?= 1
MEM ?= 2G
WALLTIME ?= 12:00:01
OPTS=-p 0 -j oe -o $(WDIR)/.pbs -q normal -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME}

SPLITDATA=${WDIR}/pipeline/1_split.py
PREPROCESS=$(WDIR)/pipeline/2_preprocess.py
FIT=$(WDIR)/pipeline/3_fit_gridsearchcv.py
EVAL=$(WDIR)/pipeline/4_eval_vae.py

J ?= 0-1:2
EP ?= os # os pfs
FLAGS ?= # empty by default

0:
	find $(WDIR)/.pbs -type f -delete

1:
	qsub ${OPTS} -J ${J} -N Split_Data -v args="${SPLITDATA}" ${WDIR}/run.sh

2:
	qsub ${OPTS} -J ${J} -N Process -v args="${PREPROCESS} --endpoint ${EP} ${FLAGS}" ${WDIR}/run.sh # IMPT: modify run.sh

3:
	qsub ${OPTS} -J ${J} -N Fit_VAE -v args="${FIT}" ${WDIR}/run.sh

4:
	qsub $(OPTS) -N Eval_VAE -v args="$(EVAL)" ${WDIR}/run.sh