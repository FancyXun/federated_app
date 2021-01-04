import numpy as np
import os
import glob

clients_url = "/tmp/model_weights/aggClients.txt"
root_url = "/tmp/model_weights/"
root_mean_url = "/tmp/model_weights/average"

if not os.path.exists(root_mean_url):
    os.makedirs(root_mean_url)
all_npz = []
with open(clients_url, "r") as f:
    c_ids = f.readlines()

for c_id in c_ids:
    pt = os.path.join(root_url, c_id.replace("\n", "").strip(" "))
    all_npz.append(glob.glob(pt+"/*.npz"))

all_npz = np.asarray(all_npz)
all_npz = all_npz.transpose()
for npz_s in all_npz:
    list_of_array = []
    mean_layer_name = ""
    for npz in npz_s:
        weights = np.load(npz)
        mean_layer_name = npz.split("/")[-1].split(".")[0]+".txt"
        list_of_array.append(weights['layer_entry'])
    mean_arr = np.mean(list_of_array, axis=0)
    np.savetxt(os.path.join(root_mean_url, mean_layer_name), mean_arr, fmt='%1.6f')
