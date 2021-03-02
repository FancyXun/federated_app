import tensorflow as tf


in_path = "../arch/pb/mobileFaceNet_pts.pb"
out_path = "../arch/pb/mobileFaceNet_pts.tflite"

input_tensor_name = ["input"]
input_tensor_shape = {"input": [1, 112, 112, 3]}
classes_tensor_name = ["embeddings"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name,
                                                      classes_tensor_name, input_shapes=input_tensor_shape)

with open(out_path, "wb") as f:
    f.write(converter.convert())

