import numpy as np
import sys

args = sys.argv
try:
    samples = args[args.index('-n') + 1] # numero di esempi
except:
    exit(-1)
try:
    model_type = args[args.index('-m') + 1] # modello da utilizzare
except:
    exit(-1)
try:
    layers = args[args.index('-l') + 1] # layer da usare solo se metto ffnn
except:
    layers = None


np.set_printoptions(precision = 2)
# conversione delle matrici di confusione
if model_type == 'ffnn':
    concat_path = 'ffnn_C/' + str(samples) + '/' + str(layers) + '/'
else:
    concat_path = 'other_C/' + str(samples) + '/'

path = 'results/' + concat_path
for i in range(10):
    conf_mat_path = path + 'conf_mat_' + str(i+1)
    conf_mat_npy = np.round(np.load(conf_mat_path + '.npy'), decimals=3)
    np.savetxt(conf_mat_path + '.csv', conf_mat_npy, delimiter=",")

conf_mat_avg = path + 'conf_mat_avg'
conf_mat_avg_npy = np.round(np.load(conf_mat_avg + '.npy'), decimals=3)
np.savetxt(conf_mat_avg + '.csv', conf_mat_npy, delimiter=",")