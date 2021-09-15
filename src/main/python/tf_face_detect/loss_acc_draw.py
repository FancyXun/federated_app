import numpy as np
import pylab as pl


def draw_loss_acc():
    train_loss = np.loadtxt("logs_centralization/train_loss.txt")
    eval_loss = np.loadtxt("logs_centralization/eval_loss.txt")
    train_acc = np.loadtxt("logs_centralization/train_acc.txt")
    eval_acc = np.loadtxt("logs_centralization/eval_acc.txt")

    train_acc = np.asarray(
        [np.mean(train_acc[i * 10:(i + 1) * 10]) for i in range(0, train_acc.size, 10)])

    train_loss = np.asarray(
        [np.mean(train_loss[i * 10:(i + 1) * 10]) for i in range(0, train_loss.size, 10)])

    x_train_loss = np.asarray([i for i in range(train_loss.size)])
    x_eval_loss = np.asarray([i for i in range(eval_loss.size)])
    x_train_acc = np.asarray([i for i in range(train_acc.size)])
    x_eval_acc = np.asarray([i for i in range(eval_acc.size)])

    pl.plot(x_train_loss, train_loss, 'g-', label=u'tf-train-loss')
    pl.legend()
    pl.show()

    pl.plot(x_eval_loss, eval_loss, 'r-', label=u'tf-eval-loss')
    pl.legend()
    pl.show()

    pl.plot(x_train_acc, train_acc, 'g-', label=u'tf-train-acc')
    pl.legend()
    pl.show()

    pl.plot(x_eval_acc, eval_acc, 'r-', label=u'tf-eval-acc')
    pl.legend()
    pl.show()


def draw_loss_acc_distribute(client_id):
    train_loss = np.loadtxt("logs_distribution/"+client_id+"/train_loss.txt")
    eval_loss = np.loadtxt("logs_distribution/"+client_id+"/eval_loss.txt")
    train_acc = np.loadtxt("logs_distribution/"+client_id+"/train_acc.txt")
    eval_acc = np.loadtxt("logs_distribution/"+client_id+"/eval_acc.txt")


    train_acc = np.asarray(
        [np.mean(train_acc[i * 10:(i + 1) * 10]) for i in range(0, train_acc.size, 10)])

    train_loss = np.asarray(
        [np.mean(train_loss[i * 10:(i + 1) * 10]) for i in range(0, train_loss.size, 10)])

    x_train_loss = np.asarray([i for i in range(train_loss.size)])
    x_eval_loss = np.asarray([i for i in range(eval_loss.size)])
    x_train_acc = np.asarray([i for i in range(train_acc.size)])
    x_eval_acc = np.asarray([i for i in range(eval_acc.size)])

    pl.plot(x_train_loss, train_loss, 'g-', label=u'fl-train-loss-' + client_id)
    pl.legend()
    pl.show()

    pl.plot(x_eval_loss, eval_loss, 'r-', label=u'fl-eval-loss-'+ client_id)
    pl.legend()
    pl.show()

    pl.plot(x_train_acc, train_acc, 'g-', label=u'fl-train-acc-'+ client_id)
    pl.legend()
    pl.show()

    pl.plot(x_eval_acc, eval_acc, 'r-', label=u'fl-eval-acc-'+ client_id)
    pl.legend()
    pl.show()


def draw_avg(n):
    avg_train_acc = []
    avg_train_loss = []
    avg_eval_loss = []
    avg_eval_acc = []
    for client_id in range(n):
        train_loss = np.loadtxt("logs_distribution/" + str(client_id) + "/train_loss.txt")
        eval_loss = np.loadtxt("logs_distribution/" + str(client_id) + "/eval_loss.txt")
        train_acc = np.loadtxt("logs_distribution/" + str(client_id) + "/train_acc.txt")
        eval_acc = np.loadtxt("logs_distribution/" + str(client_id) + "/eval_acc.txt")
        avg_train_acc.append([train_acc[(i + 1) * 82 - 1] for i in range(100)])
        avg_train_loss.append([train_loss[(i + 1) * 82 - 1] for i in range(100)])
        avg_eval_loss.append(eval_loss)
        avg_eval_acc.append(eval_acc)

    avg_train_acc = np.asarray(avg_train_acc)
    avg_train_acc = np.mean(avg_train_acc, axis=0)
    x_avg_train_acc = np.asarray([i for i in range(avg_train_acc.size)])
    pl.plot(x_avg_train_acc, avg_train_acc, 'g-', label=u'fl-train-acc')
    pl.legend()
    pl.show()

    avg_train_loss = np.asarray(avg_train_loss)
    avg_train_loss = np.mean(avg_train_loss, axis=0)
    x_avg_train_loss = np.asarray([i for i in range(avg_train_loss.size)])
    pl.plot(x_avg_train_loss, avg_train_loss, 'g-', label=u'fl-train-loss')
    pl.legend()
    pl.show()

    avg_eval_loss = np.asarray(avg_eval_loss)
    avg_eval_loss = np.mean(avg_eval_loss, axis=0)
    x_avg_eval_loss = np.asarray([i for i in range(avg_eval_loss.size)])
    pl.plot(x_avg_eval_loss, avg_eval_loss, 'r-', label=u'fl-eval-loss')
    pl.legend()
    pl.show()

    avg_eval_acc = np.asarray(avg_eval_acc)
    avg_eval_acc = np.mean(avg_eval_acc, axis=0)
    x_avg_eval_acc = np.asarray([i for i in range(avg_eval_acc.size)])
    pl.plot(x_avg_eval_acc, avg_eval_acc, 'r-', label=u'fl-eval-acc')
    pl.legend()
    pl.show()



def train_val_distribute():
    with open("logs_centralization/train_images.txt", "r") as f:
        train_images = [i.replace("\n", "") for i in f.readlines()]

    with open("logs_centralization/all_images.txt", "r") as f:
        all_images = [i.replace("\n", "") for i in f.readlines()]

    with open("logs_centralization/val_images.txt", "r") as f:
        val_images = [i.replace("\n", "") for i in f.readlines()]

    train_bins = [0 for i in range(1006)]
    val_bins = [0 for i in range(1006)]
    x = list(range(1006))
    for i in train_images:
        train_bins[int(i.split("/")[1])] += 1
    for i in val_images:
        val_bins[int(i.split("/")[1])] += 1

    pl.plot(x, train_bins, 'g-', label=u'train_distribute')
    pl.plot(x, val_bins, 'r-', label=u'val_distribute')
    pl.legend()
    pl.show()


draw_avg(2)
for i in range(2):
    draw_loss_acc_distribute(str(i))
# draw_loss_acc()
# train_val_distribute()
