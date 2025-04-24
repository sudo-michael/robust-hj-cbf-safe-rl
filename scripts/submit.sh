#!/bin/bash

# UTD_RATIO='1' CDR='0.01' CLN='True' IT='0.1' sbatch dropq.sh;
# UTD_RATIO=
for i in {10..50..10}; do
    UTD_RATIO='20' SEED=$i GN='cbf_backend' sbatch dropq.sh
done
