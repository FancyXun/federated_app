package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import computation.TensorEntity;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
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
import io.grpc.utils.LiteDownload;

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
        private final String path = "http://192.168.89.154:8888/images";
        private final String image_txt = "images.txt";
        private static String localId = UUID.randomUUID().toString().replaceAll("-", "");
        private static String liteModelUrl = "/data/user/0/io.grpc.computation/cache/model";


        protected LocalTraining(Activity activity, Context context) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
        }

        @Override
        protected String doInBackground(String... params) {
            // client id
            // server IP and port
            File file = new File(liteModelUrl);
            LiteDownload.downloadFile("http://52.81.162.253:8000/res/model_train.tflite", file);
            String host = "192.168.89.88";
            int port = 50051;
            ValueReply valueReply = runOneRound(host, port);
            System.out.println(valueReply.getMessage());
            return "success";
        }



        public Session init(Session session, String initName) {
            // initializer
            session.runner().addTarget(initName).run();
            return session;
        }

        @SuppressLint("SetTextI18n")
        public ValueReply runOneRound(String host, int port) {
            // todo: 是否需要流传输
            channel = ManagedChannelBuilder
                    .forAddress(host, port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ClientRequest.Builder builder = ClientRequest.newBuilder().setId(localId);
            Model model = stub.callModel(builder.build());
            Activity activity = activityReference.get();
            // the round of federated training
            int round = model.getRound();
            round = 10;
            String dataSplit = model.getMessage();

            // 网络层数
            List<Layer> layerList = model.getLayerList();
            List<Meta> metaList = model.getMetaList();

            // 获取graph的定义
            Graph graph = new Graph();
            graph.importGraphDef(model.getGraph().toByteArray());
            //
            initName = metaList.get(2).getMetaName();
            optimizerName = metaList.get(3).getMetaName();
            lossName = metaList.get(4).getMetaName();


            session = new Session(graph);
            ValueReply valueReply = null;
            for (int r = 0; r < round; r++) {
                int layer_size = layerList.size();
                // 初始化session 训练参数, 如果是第一轮,需要本地初始化进行训练
                // 否则需要把server的model weights set 到模型里面
                session = init(session, initName);
                if (r != 0) {
                    // 从sever中获取模型参数, 每一层对应一个request
                    // todo: 由于获取模型每一层的weights没有拆分,所以需要使用流传输.后续需要拆分每一层的weights
                    //  ,参考models回传逻辑
                    for (int i = 0; i < layer_size; i++) {
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
                        // 每次获得一层参数后，把参数feed到session中
                        // todo:如果发生网络故障,如何处理?
                        session.runner().feed(layer.getLayerInitName(), tensor)
                                .addTarget(layer.getLayerName() + "/Assign")
                                .run();
                    }
                }
                // 本地训练
                TextView train_loss_view = null;
                if (activity != null){
                    train_loss_view = activity.findViewById(R.id.TrainLoss);
                }
                float loss = train(train_loss_view);
                System.out.println("round " + r + ": " + loss);
                // get model weights
//                ModelWeights.Builder modelWeightsBuilder = getWeights(layerList, layer_size);
//                model = stub.callModel(builder.build());
//              ValueReply valueReply  = computeStream(stub, layerList, layer_size);
//              ValueReply valueReply = stub.computeWeights(modelWeightsBuilder.build());
                train_loss_view.setText("round " + r + ": " + loss);
                computeStream(stub, layerList, layer_size);
                valueReply = stub.computeFinish(builder.build());
            }
            return valueReply;

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

        @SuppressLint("SetTextI18n")
        public float train(TextView train_loss_view ) {
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
            /*
             */
            // 获取本地的图片列表...
            ArrayList<String> fileList = new FileUtils(context, "sampleData/casiaWebFace").getFileList();
            int batch_size = 16;
            float batch_size_loss = 0;
            float total_loss = 0;
            int height = 112;
            int width = 96;
            int channel = 3;
            int label_num = 10575;
            try {
                // todo: 考虑到大量图片无法打入app,目前折中办法使用restful获取远程图片
                InputStreamReader inputreader = new InputStreamReader(context.getAssets().open(image_txt));
                BufferedReader buffreader = new BufferedReader(inputreader);
                String line;
                int line_number = 0;
                float[][][][] x = new float[batch_size][height][width][channel];
                int batch_size_iter = 0;
                while ((line = buffreader.readLine()) != null) {
                    try {
                        URL url = new URL(path + line);
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("GET");
                        conn.setConnectTimeout(8000);
                        conn.setReadTimeout(8000);
                        conn.connect();
                        if (conn.getResponseCode() == 200) {
                            InputStream is = conn.getInputStream();
                            Bitmap bmp = BitmapFactory.decodeStream(is);
                            Mat image = new Mat();
                            Utils.bitmapToMat(bmp, image);
                            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGRA2BGR);
                            Size size = new Size(width, height);
                            Imgproc.resize(image, image, size);
                            int label = Integer.parseInt(line.split("/")[1]);
                            float[][] label_oneHot = new float[batch_size][label_num];
                            label_oneHot[batch_size_iter][label] = 1;
                            DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                            if (batch_size_iter < batch_size - 1) {
                                batch_size_iter++;
                                line_number++;
                                System.out.println(line + " " + line_number + " ");
                                continue;
                            } else {
                                batch_size_iter = 0;
                                System.out.println("------------------------------------------");
                            }

                            Session.Runner runner = session.runner()
                                    .feed("x", Tensor.create(x))
                                    .feed("y", Tensor.create(label_oneHot));
                            // bp
                            runner.addTarget(optimizerName).run();
                            // loss
                            float[] loss = new float[1];
                            Tensor train_loss = runner.fetch(lossName).run().get(0);
                            train_loss.copyTo(loss);
//                            for (int i = 0; i < batch_size; i++) {
//                                batch_size_loss = batch_size_loss + loss[i];
//                            }
//                            total_loss += (batch_size_loss / batch_size);
                            total_loss += loss[0];
                            System.out.println(line + " " + line_number + " " + loss[0]);
                            train_loss_view.setText(line + ": " + line_number + ": " + loss[0]);
                            batch_size_loss = 0;
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                total_loss = total_loss / ((float) line_number / batch_size);
            } catch (IOException e) {
                e.printStackTrace();
            }

// --------------------------------------------------------------------------------------------------
//            for (String filePath : fileList) {
//                Mat image = Imgcodecs.imread(cacheFile(filePath).getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
//                int label = Integer.parseInt(filePath.split("/")[filePath.split("/").length - 2]);
//                float[][] label_oneHot = new float[batch_size][1006];
//                label_oneHot[0][label] = 1;
//                float[][][][] x = DataConverter.cvMat_3dArray(image, batch_size);
//                Session.Runner runner = session.runner()
//                        .feed("x", Tensor.create(x))
//                        .feed("y", Tensor.create(label_oneHot));
//                // bp
//                runner.addTarget(optimizerName).run();
//                // loss
//                float[] loss = new float[batch_size];
//                Tensor train_loss = runner.fetch(lossName).run().get(0);
//                train_loss.copyTo(loss);
//                for (int i = 0; i < batch_size; i++) {
//                    batch_size_loss = batch_size_loss + loss[i];
//                }
//                System.out.println(batch_size_loss);
//                total_loss += (batch_size_loss / batch_size);
//                batch_size_loss = 0;
//            }
//            total_loss = total_loss / ((float) fileList.size() / batch_size);
//            return total_loss;
            return total_loss;
        }

        public ValueReply computeStream(ComputationGrpc.ComputationBlockingStub stub,
                                        List<Layer> layerList, int layer_size) {
            Pattern p = Pattern.compile("\\d+");
            ValueReply valueReply = null;
            // 回传参数,一层的参数可能很大,所以一层可能会有多个request
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
                // 获取某一层的参数
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
                        layerWeightsBuilder.setLayerId(i);
                        layerWeightsBuilder.setPart(part);
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
                    valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
                }
            }
            return valueReply;
        }

        @Override
        protected void onPostExecute(String result) {
            Activity activity = activityReference.get();
            if (activity == null) {
                return;
            }
            Button train_button = (Button) activity.findViewById(R.id.train);
            train_button.setEnabled(true);
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

        public File cacheFile(InputStream is) {
            File file = new File(context.getCacheDir() + "/tmp");
            try {
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
    }


}
