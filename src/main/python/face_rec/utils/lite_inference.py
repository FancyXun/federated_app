import tensorflow as tf
import numpy as np
import mxnet as mx
import pickle
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics
import time

from verification import evaluate

eval_db_path = "/Users/voyager/Downloads/MobileFaceNet_TF-master/datasets/faces_ms1m_112x112/lfw.bin"


def load_data(image_size):
    bins, issame_list = pickle.load(open(eval_db_path, 'rb'), encoding='bytes')
    datasets = np.empty((len(issame_list) * 2, image_size[0], image_size[1], 3))

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.imdecode(np.fromstring(_bin, np.uint8), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img - 127.5
        img = img * 0.0078125
        datasets[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(datasets.shape)

    return datasets, issame_list

model_file = "../graph/modelFaceNet.tflite"
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

data_set = load_data([112, 112])
emb_array = np.zeros((data_set[0].shape[0], 192), dtype=np.float32)
for i in range(len(data_set[0])):
    tmp = np.expand_dims(data_set[0][i], axis=0).astype(dtype=np.float32)
    print(time.time())
    interpreter.set_tensor(input_details[0]['index'], tmp)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    emb_array[i] = output_data
    print(time.time())
    print(output_data)
    print("-------------------")

    # Run forward pass to calculate embeddings

tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, data_set[1],
                                                 nrof_folds=10)
print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

auc = metrics.auc(fpr, tpr)
print('Area Under Curve (AUC): %1.3f' % auc)
# eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
print('Equal Error Rate (EER): %1.3f' % eer)