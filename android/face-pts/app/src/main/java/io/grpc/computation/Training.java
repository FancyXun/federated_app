package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.WeakReference;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import computation.TensorEntity;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationReply;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.learning.computation.Layer;
import io.grpc.learning.computation.LayerWeights;
import io.grpc.learning.computation.LayerWeightsRequest;
import io.grpc.learning.computation.Meta;
import io.grpc.learning.computation.Model;
import io.grpc.learning.computation.ModelWeights;
import io.grpc.learning.computation.ValueReply;
import io.grpc.task.StreamCall;
import io.grpc.utils.DataConverter;
import io.grpc.utils.FileUtils;

public class Training {
    static class LocalTraining extends StreamCall {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;
        @SuppressLint("StaticFieldLeak")
        private Context context;
        private String placeholder_x;
        private String placeholder_y;
        private String initName;
        private String optimizerName;
        private String lossName;
        private Session session;
        private final int maxFloatNumber = 1000000;


        protected LocalTraining(Activity activity, Context context) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
        }

        @Override
        protected String doInBackground(String... params) {
            // Test local training
//            localTraining("file:///android_asset/protobuffer/inception_resnet.pb");

            String localId = "123";
            String modelName = "LogisticsRegression";
            // server IP and port
            String host = "192.168.89.88";
            int port = 50051;

            ValueReply valueReply = runOneRoundStream(host, port, localId);
            System.out.println(valueReply.getMessage());
            return "success";
        }

        public void localTraining(String pbPath) {
            Graph graph = new Graph();
            InputStream modelStream = null;
            try {
                boolean var1 = pbPath.startsWith("file:///android_asset/");
                String var2 = var1 ? pbPath.split("file:///android_asset/")[1] : pbPath;
                modelStream = context.getAssets().open(var2);
                ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                int nRead;
                byte[] data = new byte[1024];
                while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                    buffer.write(data, 0, nRead);
                }
                buffer.flush();
                byte[] byteArray = buffer.toByteArray();
                graph.importGraphDef(byteArray);
                Iterator<Operation> operationIterator = graph.operations();
                Session session = new Session(graph);
                String trainable_var = "file:///android_asset/protobuffer/inception_resnet_trainable_var.txt";
                String feed_fetch_var = "file:///android_asset/protobuffer/inception_resnet_feed_fetch.txt";
                String data_path = "sampleData/casiaWebFace";
                LinkedHashMap<String, String> modelMap = loadModelMeta(trainable_var);
                LinkedHashMap<String, String> metaMap = loadModelMeta(feed_fetch_var);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public Session init(Session session, String initName) {
            // initializer
            session.runner().addTarget(initName).run();
            return session;
        }

        public void runOneRound(String host, int port, String localId) {
            channel = ManagedChannelBuilder
                    .forAddress(host, port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ClientRequest.Builder builder = ClientRequest.newBuilder().setId(localId);
            Model model = stub.callModel(builder.build());
            // the round of federated training
            int round = model.getRound();
            String dataSplit = model.getMessage();

            List<Layer> layerList = model.getLayerList();
            List<Meta> metaList = model.getMetaList();

            Graph graph = new Graph();
            // Get graph from server
            graph.importGraphDef(model.getGraph().toByteArray());
            // create session for tensorflow android
            session = new Session(graph);
            for (int i = 0; i < layerList.size(); i++) {
                Layer layer = layerList.get(i);
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
                Tensor tensor = Tensor.create(getShape(dim_count,
                        tensorProto.getTensorShape()), FloatBuffer.wrap(floatArray));
                session.runner().feed(layer.getLayerInitName(), tensor)
                        .addTarget(layer.getLayerName() + "/Assign")
                        .run();
                System.out.println(i + ";" + layerList.size());
            }
        }

        public ValueReply runOneRoundStream(String host, int port, String localId) {
            channel = ManagedChannelBuilder
                    .forAddress(host, port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ValueReply valueReply = null;
            for (int i = 0; i < 5; i++) {

                ClientRequest.Builder builder = ClientRequest.newBuilder().setId(localId);
                Model model = stub.callModel(builder.build());
                // the round of federated training
                int round = model.getRound();
                String dataSplit = model.getMessage();

                List<Layer> layerList = model.getLayerList();
                List<Meta> metaList = model.getMetaList();

                Graph graph = new Graph();
                // Get graph from server
                graph.importGraphDef(model.getGraph().toByteArray());
                // create session for tensorflow android
                session = new Session(graph);
                // get model weights
                ModelWeights modelWeights = stub.callModelWeights(builder.build());

                // todo: remove hardcore of meta list
            /*
            meta list
            1... the placeholder name of x
            2... the placeholder name of y
            3... the init name
            4... optimizer name
            5... metrics name like loss, auc
             */
                // the placeholder name of x
                initName = metaList.get(2).getMetaName();
                optimizerName = metaList.get(3).getMetaName();
                lossName = metaList.get(4).getMetaName();
                // init session
                // set weights
                int layer_size = modelWeights.getTensorCount();
                if (i ==0){
                    session = init(session, initName);
                }
                else{
                    session = init(session, initName);
                    setWeights(layerList, modelWeights);
                }
                // one round local train
                float loss = train();
                System.out.println(loss);
                // get model weights
                ModelWeights.Builder modelWeightsBuilder = getWeights(layerList, layer_size);
                model = stub.callModel(builder.build());
//              ValueReply valueReply  = computeStream(stub, layerList, layer_size);
//              ValueReply valueReply = stub.computeWeights(modelWeightsBuilder.build());
                computeStream(stub, layerList, layer_size);
                valueReply = stub.computeFinish(builder.build());
            }
            return valueReply;
        }

        public ValueReply computeStream(ComputationGrpc.ComputationBlockingStub stub,
                                        List<Layer> layerList, int layer_size) {
            Pattern p = Pattern.compile("\\d+");
            ValueReply valueReply = null;
            for (int i = 0; i < layer_size; i++) {
                LayerWeights.Builder layerWeightsBuilder = LayerWeights.newBuilder();

                TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                        TensorEntity.TensorShapeProto.newBuilder();
                Matcher m = p.matcher(layerList.get(i).getLayerShape());
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
                Tensor weights = session.runner().
                        fetch(layerList.get(i).getLayerName()).run().get(0);
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
                            layerWeightsBuilder.setLayerId(i);
                            layerWeightsBuilder.setPart(part);
                            System.out.println(layerList.get(i).getLayerName() + " " + floats.length + " " + size);
                            valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
                            j = 0;
                            size = size - maxFloatNumber;
                            part++;
                            if (size == 0) {
                                flag = false;
                            }
                            tensorBuilder.clear();
                        }
                        else{
                            j++;
                        }
                    }
                    if (flag) {
                        tensorBuilder.setTensorShape(tensorShapeBuilder);
                        layerWeightsBuilder.setTensor(tensorBuilder);
                        layerWeightsBuilder.setLayerId(i);
                        layerWeightsBuilder.setPart(part);
                        System.out.println(layerList.get(i).getLayerName() + " " + floats.length + " " + size);
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
                    layerWeightsBuilder.setLayerId(i);
                    layerWeightsBuilder.setPart(0);
                    System.out.println(layerList.get(i).getLayerName() + " " + floats.length + " " + size);
                    valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
                }
            }
            return valueReply;
        }

        public void setWeights(List<Layer> layerList, ModelWeights modelWeights) {
            for (int i = 0; i < layerList.size(); i++) {
                TensorEntity.TensorProto tensorProto = modelWeights.getTensor(i);
                Layer layer = layerList.get(i);
                List<Float> floatList = tensorProto.getFloatValList();
                float[] floatArray = new float[floatList.size()];
                int j = 0;
                for (Float f : floatList) {
                    floatArray[j++] = (f != null ? f : Float.NaN);
                }
                int dim_count = tensorProto.getTensorShape().getDimCount();
                Tensor tensor = Tensor.create(getShape(dim_count,
                        tensorProto.getTensorShape()), FloatBuffer.wrap(floatArray));
                session.runner().feed(layer.getLayerInitName(), tensor)
                        .addTarget(layer.getLayerName() + "/Assign")
                        .run();
            }
        }

        public ModelWeights.Builder getWeights(List<Layer> layerList, int layer_size) {
            ModelWeights.Builder modelWeightsBuilder = ModelWeights.newBuilder();
            Pattern p = Pattern.compile("\\d+");
            for (int i = 0; i < layer_size; i++) {
                TensorEntity.TensorProto.Builder tensorBuilder =
                        TensorEntity.TensorProto.newBuilder();
                TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                        TensorEntity.TensorShapeProto.newBuilder();
                Matcher m = p.matcher(layerList.get(i).getLayerShape());
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
                Tensor weights = session.runner().
                        fetch(layerList.get(i).getLayerName()).run().get(0);
                weights.writeTo(floatBuffer);
                float[] floats = floatBuffer.array();
                for (int j = 0; j < floats.length; j++) {
                    tensorBuilder.addFloatVal(floats[j]);
                }
                tensorBuilder.setTensorShape(tensorShapeBuilder);
                modelWeightsBuilder.addTensor(i, tensorBuilder);
            }
            return modelWeightsBuilder;
        }

        public long[] getShape(int dim, TensorEntity.TensorShapeProto tensorShape) {
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

        public float train() {
            int round = 10;
            // training code converter from python
            /*
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            x_batch = ...
            y_batch = ...
            feeds_train = {input_label: y_batch, input_x: x_batch}
            sess.run(optimizer, feed_dict=feeds_train)
            print(sess.run(loss, feed_dict=feeds_train))
             */
            // feed data
            ArrayList<String> fileList = new FileUtils(context, "sampleData/casiaWebFace").getFileList();
            int batch_size = 1;
            float batch_size_loss = 0;
            float total_loss = 0;
            for (String filePath : fileList) {
                Mat image = Imgcodecs.imread(cacheFile(filePath).getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
                int label = Integer.parseInt(filePath.split("/")[filePath.split("/").length - 2]);
                float[][] label_oneHot = new float[batch_size][1006];
                label_oneHot[0][label] = 1;
                float[][][][] x = DataConverter.cvMat_3dArray(image, batch_size);
                Session.Runner runner = session.runner()
                        .feed("x", Tensor.create(x))
                        .feed("y", Tensor.create(label_oneHot));
                // bp
                runner.addTarget(optimizerName).run();
                // loss
                float[] loss = new float[batch_size];
                Tensor train_loss = runner.fetch(lossName).run().get(0);
                train_loss.copyTo(loss);
                for (int i = 0; i < batch_size; i++) {
                    batch_size_loss = batch_size_loss + loss[i];
                }
                System.out.println(batch_size_loss);
                total_loss += (batch_size_loss / batch_size);
                batch_size_loss = 0;
            }
            total_loss = total_loss / ((float) fileList.size() / batch_size);
            return total_loss;
        }

        public LinkedHashMap<String, String> loadModelMeta(String filePath) {
            LinkedHashMap<String, String> map = new LinkedHashMap<String, String>();
            boolean var1 = filePath.startsWith("file:///android_asset/");
            String var2 = var1 ? filePath.split("file:///android_asset/")[1] : filePath;
            String line;
            BufferedReader reader = null;
            try {
                InputStream modelStream = context.getAssets().open(var2);
                reader = new BufferedReader(new InputStreamReader(modelStream));
                int emptyLine = 0;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(":", 2);
                    if (parts.length >= 2) {
                        String key = parts[0];
                        String value = parts[1];
                        map.put(key, value);
                    } else {
                        System.out.println("ignoring line: " + line);
                        map.put(String.valueOf(emptyLine), "null");
                        emptyLine += 1;
                    }
                }

                for (String key : map.keySet()) {
                    System.out.println(key + ":" + map.get(key));
                }
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return map;
        }

        public File cacheFile(String filename) {
            File file = new File(context.getCacheDir() + "/tmp");
            try {
                InputStream is = context.getAssets().open(filename);
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();
                FileOutputStream fos = new FileOutputStream(file);
                fos.write(buffer);
                fos.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            return file;
        }
    }


}
