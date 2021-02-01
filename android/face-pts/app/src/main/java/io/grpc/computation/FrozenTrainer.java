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
import java.io.IOException;
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
        public static String server_ip = "192.168.0.102";
        public static int server_port = 50051;
        public static final String path = "http://52.81.162.253:8000/res/CASIA-WebFace-aligned";
        public static final String image_txt = "images.txt";
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
        public static int batch_size = 8;
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
                }
            }
            TextView train_loss_view = null;
            if (activity != null) {
                train_loss_view = activity.findViewById(R.id.TrainLoss);
            }
            train(train_loss_view);
            System.out.println("-------------------------round " +
                    ClientInfo.round + ": " + TrainInfo.total_loss);
            ClientInfo.local_loss = TrainInfo.total_loss;
            Timestamp timestamp = new Timestamp(System.currentTimeMillis());
            System.out.println("---------------"+timestamp);
            computeStream(stub);
            timestamp = new Timestamp(System.currentTimeMillis());
            System.out.println("----------------"+timestamp);
            stub.computeFinish(builder.build());
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
                System.out.println(op.name());
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
                    Mat image = TrainerStreamUtils.getImage(ServeInfo.path + line, imageInfo);
                    int label = Integer.parseInt(line.split("/")[1]);
                    label_oneHot[batch_size_iter][label] = 1;
                    assert image != null;
                    try{
                        DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                    }
                    catch (NullPointerException e){
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
                    runner
                            .feed("input_x", Tensor.create(x))
                            .feed("input_y", Tensor.create(label_oneHot))
                            .feed(MetaInfo.learningRate, Tensor.create(0.0001f))
                            .addTarget(MetaInfo.optimizerName)
                            .run();


                    List<Tensor<?>> fetched_tensors = runner
//                            .feed("input_x", Tensor.create(x))
//                            .feed("input_y", Tensor.create(label_oneHot))
                            .fetch(MetaInfo.lossName)
                            .fetch(MetaInfo.accName)
                            .run();

                    System.out.println("-----" + ": " + line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                            " acc: " + fetched_tensors.get(1).floatValue());
                    label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
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
            }
        }
    }
}
