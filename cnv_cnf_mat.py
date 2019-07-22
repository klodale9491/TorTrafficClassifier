import numpy as np
import sys, os

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

try:
    avg = args[args.index('--avg')] # only average mat
except:
    avg = 0


def save_dir(filedir, savedir):
    
    def save_csv(filename, savedir, suffix = ""):
        conf_mat_npy_path = os.path.join(filedir, filename) # npy file
        conf_mat_npy = np.load(conf_mat_npy_path + '.npy') # npy obj
        csv_file_path = os.path.join(savedir, filename + suffix + '.csv') # csv file
        if os.path.exists(csv_file_path): # overwrite
            os.remove(csv_file_path)
        np.savetxt(csv_file_path, conf_mat_npy, delimiter=",", fmt='%2.1f')
    
    num_mat = 10
    avg_mat_name = 'conf_mat_avg'

    if not avg:

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for i in range(num_mat + 1):

            filename = 'conf_mat_' + str(i+1)

            if i == num_mat: #avg
                filename = avg_mat_name

            save_csv(filename, savedir)
    else:
        basedir, tail = os.path.split(savedir)
        avgroot = os.path.join(basedir, "avg")
        if not os.path.exists(avgroot):
            os.makedirs(avgroot)
        save_csv(avg_mat_name, avgroot, "-" + tail)

root = "results"
saveroot = os.path.join(root, "csv")

if model_type == 'ffnn':
    modeldir = "ffnn_C"
    base_dir = os.path.join(modeldir, str(samples))

    filedir = os.path.join(root, base_dir, str(layers))
    savedir = os.path.join(saveroot, base_dir, str(layers))

    save_dir(filedir, savedir)

elif model_type == 'other':
    modeldir = "other_C"
    base_dir_conv = os.path.join(modeldir, "convolution", str(samples))
    base_dir_org = os.path.join(modeldir, "original", str(samples))

    filedir_conv = os.path.join(root, base_dir_conv)
    filedir_org = os.path.join(root, base_dir_org)

    savedir_conv = os.path.join(saveroot, base_dir_conv)
    savedir_org = os.path.join(saveroot, base_dir_org)  

    for subdir in os.listdir(filedir_conv):
        save_dir(os.path.join(filedir_conv, subdir), os.path.join(savedir_conv, subdir))
         
    for subdir in os.listdir(filedir_org):
        save_dir(os.path.join(filedir_org, subdir), os.path.join(savedir_org, subdir))