package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.TaskInfo;
import android.content.Context;
import android.widget.TextView;

import org.opencv.core.Mat;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.Certificate;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.Layer;
import io.grpc.learning.computation.LayerFeed;
import io.grpc.learning.computation.Meta;
import io.grpc.learning.computation.Model;
import io.grpc.learning.computation.TrainMetrics;
import io.grpc.learning.computation.ValueReply;
import io.grpc.transmit.StreamCall;
import io.grpc.utils.DataConverter;
import io.grpc.utils.Timer;
import io.grpc.vo.ImageInfo;

public class FrozenTrainer {

    static class MetaInfo {
        public static String initName;
        public static String optimizerName;
        public static String lossName;
        public static String learningRate;
        public static String accName;
    }

    static class ServeInfo {
        public static String server_ip = "192.168.89.249";
        public static int server_port = 50051;
        public static final String path = "http://52.81.162.253:8000/res/CASIA-WebFace-aligned";
        public static final String image_txt = "train_images_0.txt";
        public static final String image_val_txt = "val_images_0.txt";
    }

    static class ClientInfo {
        public static String localId =
                UUID.randomUUID().toString().replaceAll("-", "");
        public static int round = 0;
        public static String token = null;
        public static float local_loss = Float.MAX_VALUE;
        public static float loss_threshold = 0.01f;
        public static boolean firstRound = true;
    }

    static class TrainInfo {
        public static int batch_size = 16;
        public static float total_loss = 0;
    }

    static class LocalTraining extends StreamCall {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;
        @SuppressLint("StaticFieldLeak")
        private Context context;
        private Session session;
        //
        private List<Layer> layerList;
        //
        private List<LayerFeed> layerFeedList;
        //
        private List<Layer> trainableLayerList;
        public  ArrayList<Operation> fetch_ops = new ArrayList();

        public ArrayList<Float> train_loss_list = new ArrayList<>();
        public ArrayList<Float> val_loss_list = new ArrayList<>();
        public ArrayList<Float> train_acc_list = new ArrayList<>();
        public ArrayList<Float> val_acc_list = new ArrayList<>();



        protected LocalTraining(Activity activity, Context context) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
        }

        /**
         * @param params
         * @return
         */
        @Override
        protected String doInBackground(String... params) {
            // For test
//            Activity activity = activityReference.get();
//            TextView train_loss_view = null;
//            if (activity != null) {
//                train_loss_view = activity.findViewById(R.id.TrainLoss);
//            }
//            if (true){
//                local_train(train_loss_view);
//            }
            channel = ManagedChannelBuilder
                    .forAddress(ServeInfo.server_ip, ServeInfo.server_port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            while (ClientInfo.local_loss > ClientInfo.loss_threshold) {
                ClientRequest.Builder builder = ClientRequest.newBuilder().setId(ClientInfo.localId);
                Certificate certificate = stub.callTraining(builder.build());
                if (ClientInfo.token == null) {
                    ClientInfo.token = certificate.getToken();
                } else {
                    while (ClientInfo.token.equals(certificate.getToken())) {
                        Timer.sleep(3000);
                        certificate = stub.callTraining(builder.build());
                    }
                }
                while (!certificate.getServerState().equals("ready")) {
                    Timer.sleep(3000);
                    certificate = stub.callTraining(builder.build());
                }
                ClientInfo.token = certificate.getToken();
                runOneRound(stub, builder);
                session.close();
                ClientInfo.round += 1;
            }
            return "Training Finished!";
        }

        /**
         * @param session
         * @param initName
         * @return
         */
        public Session initSession(Session session, String initName) {
            session.runner().addTarget(initName).run();
            return session;
        }

        /**
         * @param stub
         * @param builder callModel ->
         */
        @SuppressLint("CheckResult")
        public void runOneRound(ComputationGrpc.ComputationBlockingStub stub,
                                ClientRequest.Builder builder) {
            Model model = stub.callModel(builder.build());
            Activity activity = activityReference.get();
            ModelGraphInit(model);
            session = initSession(session, MetaInfo.initName);
            if (!ClientInfo.firstRound) {
                for (int i = 0; i < layerFeedList.size(); i++) {
                    LayerFeed layerFeed = layerFeedList.get(i);
                    Tensor tensor = TrainerStreamUtils.getLayerWeightsByName(ClientInfo.localId,
                            layerFeed.getLayerFeedWeightsName(), stub);
                    session.runner().feed(layerFeed.getLayerInitFeedWeightsName(),
                            tensor).addTarget(layerFeed.getLayerFeedWeightsName() + "/Assign").run();
                    System.out.println(i + ":" + layerFeed.getLayerFeedWeightsName());
                    tensor.close();
                }
            }
            TextView train_loss_view = null;
            if (activity != null) {
                train_loss_view = activity.findViewById(R.id.TrainLoss);
            }
            train(train_loss_view);
            eval(train_loss_view);
            for (int i =0 ; i< train_loss_list.size();i ++){
                TrainInfo.total_loss = TrainInfo.total_loss + train_loss_list.get(i);
            }
            ClientInfo.local_loss = TrainInfo.total_loss/train_loss_list.size();
            computeStream(stub);
            TrainMetrics.Builder trainBuilder =  TrainMetrics.newBuilder();
            trainBuilder.addAllAccValue(train_acc_list);
            trainBuilder.addAllLossValue(train_loss_list);
            trainBuilder.addAllEvalAccValue(val_acc_list);
            trainBuilder.addAllEvalLossValue(val_loss_list);
            trainBuilder.setId(ClientInfo.localId);
            trainBuilder.setRound(ClientInfo.round);
            stub.computeMetrics(trainBuilder.build());
            stub.computeFinish(builder.build());
            train_acc_list.clear();
            train_loss_list.clear();
            val_acc_list.clear();
            val_loss_list.clear();
        }

        public void ModelGraphInit(Model model) {
            // check round is the first or not
            ClientInfo.firstRound = model.getFirstRound();
            // all layers
            layerList = model.getLayerList();
            // the layers which weights need to be feed  before training
            layerFeedList = model.getLayerFeedList();
            List<Meta> metaList = model.getMetaList();
            MetaInfo.initName = metaList.get(2).getMetaName();
            MetaInfo.optimizerName = metaList.get(3).getMetaName();
            MetaInfo.lossName = metaList.get(4).getMetaName();
            MetaInfo.learningRate = metaList.get(5).getMetaName();
            MetaInfo.accName = metaList.get(6).getMetaName();
            // the trainable layers in current round
            trainableLayerList = new ArrayList<>();
            for (Layer layer : layerList) {
                if (!layer.getLayerName().equals("non_trainable")) {
                    trainableLayerList.add(layer);
                }
            }
            Graph graph = new Graph();
            graph.importGraphDef(model.getGraph().toByteArray());
            for (Iterator<Operation> it = graph.operations(); it.hasNext(); ) {
                Operation op = it.next();
                if (op.name().equals(MetaInfo.lossName)){
                    fetch_ops.add(op);
                }
                if (op.name().equals(MetaInfo.accName)){
                    fetch_ops.add(op);
                }
            }
            session = new Session(graph);
        }

        @SuppressLint("SetTextI18n")
        public void train(TextView train_loss_view) {

            ImageInfo imageInfo = new ImageInfo();
            try {
                // todo: get images from assets
                InputStreamReader inputReader = new InputStreamReader(context.getAssets().open(ServeInfo.image_txt));
                BufferedReader buffReader = new BufferedReader(inputReader);
                String line;
                int line_number = 0;
                float[][][][] x = new float[TrainInfo.batch_size][imageInfo.getHeight()]
                        [imageInfo.getWidth()][imageInfo.getChannel()];
                int batch_size_iter = 0;

                int[][] label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                for (int epoch = 0; epoch < 20; epoch++) {
                while ((line = buffReader.readLine()) != null) {
                    try{
                    Mat image = TrainerStreamUtils.getImage(ServeInfo.path + line, imageInfo);
                    int label = Integer.parseInt(line.split("/")[1]);
                    label_oneHot[batch_size_iter][label] = 1;
                    assert image != null;
                    DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                    }
                    catch (Exception e){
                        e.printStackTrace();
                        continue;
                    }
                    if (batch_size_iter < TrainInfo.batch_size - 1) {
                        batch_size_iter++;
                        line_number++;
                        continue;
                    } else {
                        batch_size_iter = 0;
                        line_number++;
                    }

                    Session.Runner runner = session.runner();
                    Tensor x_t = Tensor.create(x);
                    Tensor label_oneHot_t = Tensor.create(label_oneHot);
                    Tensor lr_t = Tensor.create(0.0001f);
                    runner
                            .feed("input_x", x_t)
                            .feed("input_y", label_oneHot_t)
                            .feed(MetaInfo.learningRate, lr_t)
                            .addTarget(MetaInfo.optimizerName)
                            .run();


                    List<Tensor<?>> fetched_tensors = runner
                            .fetch(MetaInfo.lossName)
                            .fetch(MetaInfo.accName)
                            .run();

                    System.out.println("-----" + ": " + line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                            " acc: " + fetched_tensors.get(1).floatValue());
                    train_loss_view.setText(line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                            " acc: " + fetched_tensors.get(1).floatValue());
                    train_loss_list.add(fetched_tensors.get(0).floatValue());
                    train_acc_list.add(fetched_tensors.get(1).floatValue());
                    label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                    x_t.close();
                    label_oneHot_t.close();
                    lr_t.close();
                }
            }
                } catch(IOException e){
                    e.printStackTrace();
                }

        }

        @SuppressLint("SetTextI18n")
        public void eval(TextView train_loss_view) {

            ImageInfo imageInfo = new ImageInfo();
            try {
                // todo: get images from assets
                InputStreamReader inputReader = new InputStreamReader(context.getAssets().open(ServeInfo.image_val_txt));
                BufferedReader buffReader = new BufferedReader(inputReader);
                String line;
                int line_number = 0;
                float[][][][] x = new float[TrainInfo.batch_size][imageInfo.getHeight()]
                        [imageInfo.getWidth()][imageInfo.getChannel()];
                int batch_size_iter = 0;

                int[][] label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                for (int epoch = 0; epoch < 20; epoch++) {
                    while ((line = buffReader.readLine()) != null) {
                        try{
                        Mat image = TrainerStreamUtils.getImage(ServeInfo.path + line, imageInfo);
                        int label = Integer.parseInt(line.split("/")[1]);
                        label_oneHot[batch_size_iter][label] = 1;
                        assert image != null;
                        DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                        }
                        catch (Exception e){
                            e.printStackTrace();
                            continue;
                        }
                        if (batch_size_iter < TrainInfo.batch_size - 1) {
                            batch_size_iter++;
                            line_number++;
                            continue;
                        } else {
                            batch_size_iter = 0;
                            line_number++;
                        }

                        Session.Runner runner = session.runner();
                        Tensor x_t = Tensor.create(x);
                        Tensor label_oneHot_t = Tensor.create(label_oneHot);

                        List<Tensor<?>> fetched_tensors = runner
                                .feed("input_x", x_t)
                                .feed("input_y", label_oneHot_t)
                                .fetch(MetaInfo.lossName)
                                .fetch(MetaInfo.accName)
                                .run();

                        System.out.println("-----" + ": " + line_number + "eval loss: " + fetched_tensors.get(0).floatValue() +
                                "eval acc: " + fetched_tensors.get(1).floatValue());
                        train_loss_view.setText(line_number + " eval loss: " + fetched_tensors.get(0).floatValue() +
                                "eval acc: " + fetched_tensors.get(1).floatValue());
                        val_loss_list.add(fetched_tensors.get(0).floatValue());
                        val_acc_list.add(fetched_tensors.get(1).floatValue());
                        label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                        x_t.close();
                        label_oneHot_t.close();
                    }
                }
            } catch(IOException e){
                e.printStackTrace();
            }

        }

        public void computeStream(ComputationGrpc.ComputationBlockingStub stub) {
            ValueReply valueReply = null;
            TrainerStreamUtils trainerStreamUtils = new TrainerStreamUtils();
            for (Layer layer : trainableLayerList) {
                Tensor weights = session.runner().
                        fetch(layer.getLayerName()).run().get(0);
                ClientRequest.Builder clientRequestBuilder = ClientRequest.newBuilder();
                clientRequestBuilder.setToken(ClientInfo.token);
                clientRequestBuilder.setId(ClientInfo.localId);
                valueReply = trainerStreamUtils.callLayerWeights(clientRequestBuilder,
                        layer.getLayerName(), stub, weights,
                        layer.getLayerTrainableShape());
                weights.close();
            }
        }

        @SuppressLint("SetTextI18n")
        public void local_train(TextView train_loss_view) {
            Graph graph = new Graph();
            try {
                String var2 = "train/sphere_unfrozen.pb";
                InputStream modelStream = context.getAssets().open(var2);
                ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                int nRead;
                byte[] data = new byte[1024];
                while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                    buffer.write(data, 0, nRead);
                }
                buffer.flush();
                byte[] byteArray = buffer.toByteArray();
                graph.importGraphDef(byteArray);
            } catch (IOException e) {
                e.printStackTrace();
            }
            session = new Session(graph);
            session.runner().addTarget("init").run();
            ImageInfo imageInfo = new ImageInfo();
            try {
                // todo: get images from assets
                InputStreamReader inputReader = new InputStreamReader(context.getAssets().open(ServeInfo.image_txt));
                BufferedReader buffReader = new BufferedReader(inputReader);
                String line;
                int line_number = 0;
                float[][][][] x = new float[TrainInfo.batch_size][imageInfo.getHeight()]
                        [imageInfo.getWidth()][imageInfo.getChannel()];
                int batch_size_iter = 0;

                int[][] label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                for (int epoch = 0; epoch < 20; epoch++) {
                    while ((line = buffReader.readLine()) != null) {
                        try{
                            Mat image = TrainerStreamUtils.getImage(ServeInfo.path + line, imageInfo);
                            int label = Integer.parseInt(line.split("/")[1]);
                            label_oneHot[batch_size_iter][label] = 1;
                            assert image != null;
                            DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                        }
                        catch (Exception e){
                            e.printStackTrace();
                            continue;
                        }
                        if (batch_size_iter < TrainInfo.batch_size - 1) {
                            batch_size_iter++;
                            line_number++;
                            continue;
                        } else {
                            batch_size_iter = 0;
                            line_number++;
                        }

                        Session.Runner runner = session.runner();
                        Tensor x_t = Tensor.create(x);
                        Tensor label_oneHot_t = Tensor.create(label_oneHot);
                        Tensor lr_t = Tensor.create(0.0001f);
                        runner
                                .feed("input_x", x_t)
                                .feed("input_y", label_oneHot_t)
                                .feed("lr:0", lr_t)
                                .addTarget("Momentum")
                                .run();


                        List<Tensor<?>> fetched_tensors = runner
                                .fetch("Mean:0")
                                .fetch("Mean_1:0")
                                .run();

                        System.out.println("-----" + ": " + line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                                " acc: " + fetched_tensors.get(1).floatValue());
                        train_loss_view.setText(line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                                " acc: " + fetched_tensors.get(1).floatValue());
                        label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                        x_t.close();
                        label_oneHot_t.close();
                        lr_t.close();
                    }
                }
            } catch(IOException e){
                e.printStackTrace();
            }

        }
    }
}
