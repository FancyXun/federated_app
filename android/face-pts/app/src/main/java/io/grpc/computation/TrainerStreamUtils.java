package io.grpc.computation;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import computation.TensorEntity;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.LayerWeights;
import io.grpc.learning.computation.LayerWeightsRequest;
import io.grpc.learning.computation.ValueReply;
import io.grpc.vo.ImageInfo;

public class TrainerStreamUtils {

    Pattern p = Pattern.compile("\\d+");

    /**
     * @param localId device id
     * @param i       layer index
     * @param stub    ComputationGrpc.ComputationBlockingStu
     */
    public static Tensor getLayerWeights(String localId, int i, ComputationGrpc.ComputationBlockingStub stub) {
        LayerWeightsRequest.Builder layerBuilder = LayerWeightsRequest.newBuilder();
        layerBuilder.setId(localId);
        layerBuilder.setLayerId(i);
        LayerWeights layerWeights = stub.callLayerWeights(layerBuilder.build());
        TensorEntity.TensorProto tensorProto = layerWeights.getTensor();
        List<Float> floatList = tensorProto.getFloatValList();
        float[] floatArray = new float[floatList.size()];
        int j = 0;
        for (Float f : floatList) {
            floatArray[j++] = (f != null ? f : Float.NaN);
        }

        int dim_count = tensorProto.getTensorShape().getDimCount();
        return Tensor.create(getShape(dim_count,
                tensorProto.getTensorShape()), FloatBuffer.wrap(floatArray));
    }


    public static Tensor getLayerWeightsByName(String localId, String layer_name, ComputationGrpc.ComputationBlockingStub stub) {
        LayerWeightsRequest.Builder layerBuilder = LayerWeightsRequest.newBuilder();
        layerBuilder.setId(localId);
        layerBuilder.setLayerName(layer_name);
        LayerWeights layerWeights = stub.callLayerWeights(layerBuilder.build());
        TensorEntity.TensorProto tensorProto = layerWeights.getTensor();
        List<Float> floatList = tensorProto.getFloatValList();
        float[] floatArray = new float[floatList.size()];
        int j = 0;
        for (Float f : floatList) {
            floatArray[j++] = (f != null ? f : Float.NaN);
        }

        int dim_count = tensorProto.getTensorShape().getDimCount();
        if (floatArray.length == 359296){
            System.out.println(tensorProto.getTensorShape());
        }
        return Tensor.create(getShape(dim_count,
                tensorProto.getTensorShape()), FloatBuffer.wrap(floatArray));
    }

    /**
     *
     * @param dim tensor dimension numbers
     * @param tensorShape the shape of tensor
     * @return 1-d array of long type
     */
    public static long[] getShape(int dim, TensorEntity.TensorShapeProto tensorShape) {
        if (dim == 1) {
            return new long[]{tensorShape.getDim(0).getSize()};
        } else if (dim == 2) {
            return new long[]{tensorShape.getDim(0).getSize(), tensorShape.getDim(1).getSize()};
        } else if (dim == 3) {
            return new long[]{tensorShape.getDim(0).getSize(), tensorShape.getDim(1).getSize(),
                    tensorShape.getDim(2).getSize()};
        } else {
            return new long[]{tensorShape.getDim(0).getSize(), tensorShape.getDim(1).getSize(),
                    tensorShape.getDim(2).getSize(), tensorShape.getDim(3).getSize()};
        }
    }

    /**
     *
     * @param ImgURL image url in server
     * @param imageInfo image information contains height, width...
     * @return Mat
     */
    public static Mat getImage(String ImgURL, ImageInfo imageInfo) {
        Mat image = new Mat();
        try {
            URL url = new URL(ImgURL);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(8000);
            conn.setReadTimeout(8000);
            conn.connect();
            if (conn.getResponseCode() == 200) {
                InputStream is = conn.getInputStream();
                Bitmap bmp = BitmapFactory.decodeStream(is);
                Utils.bitmapToMat(bmp, image);
                Imgproc.cvtColor(image, image, Imgproc.COLOR_RGBA2RGB);
                Size size = new Size(imageInfo.getWidth(), imageInfo.getHeight());
                Imgproc.resize(image, image, size);
//                bmp.recycle();
                return image;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     *
     * @param layer_name layer index
     * @param stub ComputationGrpc.ComputationBlockingStub
     * @param weights layer weights
     * @param layerShape the shape of layer
     * @return ValueReply
     */
    public ValueReply callLayerWeights(ClientRequest.Builder clientRequestBuilder,
            String layer_name, ComputationGrpc.ComputationBlockingStub stub,
                                       Tensor weights, String layerShape) {
        int maxFloatNumber = 1000000;
        ValueReply valueReply = null;
        LayerWeights.Builder layerWeightsBuilder = LayerWeights.newBuilder();
        TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                TensorEntity.TensorShapeProto.newBuilder();
        Matcher m = p.matcher(layerShape);
        int dim_index = 0;
        int size = 1;
        while (m.find()) {
            int dim = Integer.parseInt(m.group());
            size = size * dim;
            TensorEntity.TensorShapeProto.Dim.Builder dimBuilder =
                    TensorEntity.TensorShapeProto.Dim.newBuilder();
            dimBuilder.setSize(dim);
            tensorShapeBuilder.addDim(dim_index, dimBuilder);
            dim_index++;
        }
        FloatBuffer floatBuffer = FloatBuffer.allocate(size);
        weights.writeTo(floatBuffer);
        float[] floats = floatBuffer.array();
        if (size > maxFloatNumber) {
            int j = 0;
            boolean flag = true;
            TensorEntity.TensorProto.Builder tensorBuilder = null;
            int part = 0;
            while (j < size) {
                if (j == 0) {
                    tensorBuilder =
                            TensorEntity.TensorProto.newBuilder();
                }
                tensorBuilder.addFloatVal(floats[j]);
                if (j == maxFloatNumber - 1) {
                    tensorBuilder.setTensorShape(tensorShapeBuilder);
                    layerWeightsBuilder.setTensor(tensorBuilder);
                    layerWeightsBuilder.setLayerName(layer_name);
                    layerWeightsBuilder.setPart(part);
                    layerWeightsBuilder.setClientRequest(clientRequestBuilder.build());
                    valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
                    j = 0;
                    size = size - maxFloatNumber;
                    part++;
                    if (size == 0) {
                        flag = false;
                    }
                    tensorBuilder.clear();
                } else {
                    j++;
                }
            }
            if (flag) {
                tensorBuilder.setTensorShape(tensorShapeBuilder);
                layerWeightsBuilder.setTensor(tensorBuilder);
                layerWeightsBuilder.setLayerName(layer_name);
                layerWeightsBuilder.setPart(part);
                layerWeightsBuilder.setClientRequest(clientRequestBuilder.build());
                valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
            }
        } else {
            TensorEntity.TensorProto.Builder tensorBuilder =
                    TensorEntity.TensorProto.newBuilder();
            for (int j = 0; j < floats.length; j++) {
                tensorBuilder.addFloatVal(floats[j]);
            }
            tensorBuilder.setTensorShape(tensorShapeBuilder);
            layerWeightsBuilder.setTensor(tensorBuilder);
            layerWeightsBuilder.setLayerName(layer_name);
            layerWeightsBuilder.setPart(0);
            layerWeightsBuilder.setClientRequest(clientRequestBuilder.build());
            valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
        }
        weights.close();
        return valueReply;
    }
}
