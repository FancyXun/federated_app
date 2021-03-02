import os

import mxnet as mx
from PIL import Image
from mxnet import recordio
from tqdm import tqdm

path_imgidx = '/Users/voyager/Downloads/faces_emore/train.idx'
path_imgrec = '/Users/voyager/Downloads/faces_emore/train.rec'

save_img = '/Users/voyager/tensorflow_datasets/faces_ms1m_112x112'

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

# %% 1 ~ 3804847
for i in tqdm(range(3804846)):
    header, s = recordio.unpack(imgrec.read_idx(i + 1))
    img = mx.image.imdecode(s).asnumpy()
    if not os.path.exists(save_img + "/" + str(int(header.label))):
        os.makedirs(save_img + "/" + str(int(header.label)))
    im = Image.fromarray(img)
    im.save(save_img + "/" + str(int(header.label)) + "/" + str(i) + ".jpg")

    # plt.imshow(img)
    # plt.title('id=' + str(i) + 'label=' + str(header.label))
    # plt.pause(0.1)
