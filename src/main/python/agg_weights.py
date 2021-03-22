import numpy as np
import os
import glob
import sys


root_path = sys.argv[1]
clients_url = root_path + "/aggClients.txt"
root_mean_url = root_path + "/average"

if not os.path.exists(root_mean_url):
    os.makedirs(root_mean_url)
all_npz = []
with open(clients_url, "r") as f:
    c_ids = f.readlines()

for c_id in c_ids:
    pt = os.path.join(root_path, c_id.replace("\n", "").strip(" "))
    all_npz.append(glob.glob(pt+"/*.npz"))

all_npz = np.asarray(all_npz)
all_npz = all_npz.transpose()

npz_name = []
for npz_s in all_npz:
    list_of_array = []
    mean_layer_name = ""
    for npz in npz_s:
        weights = np.load(npz)
        mean_layer_name = npz.split("/")[-1].split(".")[0]+".npz"
        list_of_array.append(weights['layer_entry'])
    mean_arr = np.mean(list_of_array, axis=0)
    npz_name.append(os.path.join(root_mean_url, mean_layer_name))
    # np.savetxt(os.path.join(root_mean_url, mean_layer_name), mean_arr, fmt='%1.6f')
    np.savez(os.path.join(root_mean_url, mean_layer_name), mean_arr)

name_block = {}

for i in npz_name:
    name = i.split("__")[0]
    block = i.split("__")[1]
    if name not in name_block.keys():
        name_block[name] = []
    name_block[name].append(block)

for i in name_block.keys():
    v = name_block.get(i)
    a = np.load(i+"__"+v[0])['arr_0']
    for j in v[1:]:
        a = np.concatenate((a, np.load(i+"__"+j)['arr_0']))
    np.savetxt(i+".txt", a, fmt='%1.6f')

