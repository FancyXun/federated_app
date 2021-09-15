import sched
import time
from datetime import datetime
import numpy as np
import os
import glob

# 初始化sched模块的 scheduler 类
# 第一个参数是一个可以返回时间戳的函数，第二个参数可以在定时未到达之前阻塞。
schedule = sched.scheduler(time.time, time.sleep)
# 被周期性调度触发的函数


def printTime(inc):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    schedule.enter(inc, 0, printTime, (inc,))


# 默认参数60s
def main(inc=60):
    # enter四个参数分别为：间隔事件、优先级（用于同时间到达的两个事件同时执行时定序）、被调用触发的函数，
    # 给该触发函数的参数（tuple形式）
    schedule.enter(0, 0, printTime, (inc,))
    schedule.run()


def agg_weights():

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
        all_npz.append(glob.glob(pt + "/*.npz"))

    all_npz = np.asarray(all_npz)
    all_npz = all_npz.transpose()

    npz_name = []
    for npz_s in all_npz:
        list_of_array = []
        mean_layer_name = ""
        for npz in npz_s:
            weights = np.load(npz)
            mean_layer_name = npz.split("/")[-1].split(".")[0] + ".npz"
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
        a = np.load(i + "__" + v[0])['arr_0']
        for j in v[1:]:
            a = np.concatenate((a, np.load(i + "__" + j)['arr_0']))
        np.savetxt(i + ".txt", a, fmt='%1.6f')


# 10s 输出一次
main(10)
