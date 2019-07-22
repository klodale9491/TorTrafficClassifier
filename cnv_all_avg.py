import os
import array as a

samples = [1000,5000,10000,25000]
layers = [5,10,20,25]

for smp in samples:
    for lr in layers:
        os.system("python cnv_cnf_mat.py --avg -m ffnn -n " + str(smp) + " -l " + str(lr))

for smp in samples:
        os.system("python cnv_cnf_mat.py --avg -m other -n " + str(smp))