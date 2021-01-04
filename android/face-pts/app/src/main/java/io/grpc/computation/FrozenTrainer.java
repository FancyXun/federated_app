package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.widget.TextView;

import org.opencv.core.Mat;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.Certificate;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.Layer;
import io.grpc.learning.computation.Meta;
import io.grpc.learning.computation.Model;
import io.grpc.learning.computation.ValueReply;
import io.grpc.transmit.StreamCall;
import io.grpc.utils.DataConverter;
import io.grpc.vo.ImageInfo;

public class FrozenTrainer {
    static class LocalTraining extends StreamCall {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;
        @SuppressLint("StaticFieldLeak")
        private Context context;
        private String initName;
        private String optimizerName;
        private String lossName;
        private String dataSplit;
        private HashMap<String, Tensor> modelTrainableWeighs = new HashMap<>();
        private HashMap<String, String> modelTrainableInit = new HashMap<>();
        private boolean epochFinished = false;
        private Session session;
        private final int maxFloatNumber = 1000000;
        private String server_ip = "192.168.0.103";
        private int server_port = 50051;
        private final String path = "http://52.81.162.253:8000/res/CASIA-WebFace-aligned"; // image url
        private final String image_txt = "images.txt"; //train images
        private static String localId = UUID.randomUUID().toString().replaceAll("-", "");
        private List<Layer> layerList;
        private List<Layer> trainableLayerList;
        private int round = 0;
        private String token = null;
        private float local_loss = Float.MAX_VALUE;
        private boolean firstRound = true;


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
                    .forAddress(server_ip, server_port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            while (local_loss > 0.01) {
                ClientRequest.Builder builder = ClientRequest.newBuilder().setId(localId);
                Certificate certificate = stub.callTraining(builder.build());
                if (token == null) {
                    token = certificate.getToken();
                } else {
                    while (token.equals(certificate.getToken())) {
                        try {
                            Thread.sleep(3000);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        certificate = stub.callTraining(builder.build());
                    }
                }
                while (!certificate.getServerState().equals("ready")) {
                    certificate = stub.callTraining(builder.build());
                }
                runOneRound(stub, builder);
                round +=1;
            }
            return "Training Finished!";
        }

        /**
         * @param session
         * @param initName
         * @return
         */
        public Session init(Session session, String initName) {
            session.runner().addTarget(initName).run();
            return session;
        }

        /**
         * @param stub
         * @param builder
         */
        public void runOneRound(ComputationGrpc.ComputationBlockingStub stub,
                                ClientRequest.Builder builder) {
            Model model = stub.callModel(builder.build());
            Activity activity = activityReference.get();
            getModelGraph(model);
            model = stub.callModel(builder.build());
            activity = activityReference.get();
            getModelGraph(model);
            int layer_size = trainableLayerList.size();
            session = init(session, initName);
            if (!firstRound) {
                System.out.println("***");
                for (int i = 0; i < layer_size; i++) {
                    Layer layer = trainableLayerList.get(i);
                    session.runner().feed(layer.getLayerInitName(),
                            TrainerStreamUtils.getLayerWeights(localId, i, stub))
                            .addTarget(layer.getLayerName() + "/Assign")
                            .run();
                }
//                if (epochFinished) {
//                    for (int i = 0; i < layer_size; i++) {
//                        Layer layer = layerList.get(i);
//                        session.runner().feed(layer.getLayerInitName(),
//                                TrainerStreamUtils.getLayerWeights(localId, i, stub))
//                                .addTarget(layer.getLayerName() + "/Assign")
//                                .run();
//                    }
//                } else {
//
//                    for (String key : modelTrainableWeighs.keySet()) {
//                        session.runner().feed(key, modelTrainableWeighs.get(key))
//                                .addTarget(modelTrainableInit.get(key) + "/Assign")
//                                .run();
//                    }
//                }
            }
            TextView train_loss_view = null;
            if (activity != null) {
                train_loss_view = activity.findViewById(R.id.TrainLoss);
            }
            float loss = train(train_loss_view);
            assert train_loss_view != null;
            train_loss_view.setText("round " + round + ": " + loss);
            local_loss = loss;
            computeStream(stub, layerList, layer_size);
            stub.computeFinish(builder.build());
        }

        public void getModelGraph(Model model) {
            firstRound = model.getFirstRound();
            dataSplit = model.getMessage();
            layerList = model.getLayerList();
            List<Meta> metaList = model.getMetaList();
            trainableLayerList = new ArrayList<>();
            for (Layer layer : layerList){
                if (!layer.getLayerName().equals("non_trainable")){
                    trainableLayerList.add(layer);
                }
            }
            Graph graph = new Graph();
            graph.importGraphDef(model.getGraph().toByteArray());
            initName = metaList.get(2).getMetaName();
            optimizerName = metaList.get(3).getMetaName();
            lossName = metaList.get(4).getMetaName();
            session = new Session(graph);
        }

        @SuppressLint("SetTextI18n")
        public float train(TextView train_loss_view) {
            int batch_size = 16;
            float total_loss = 0;
            ImageInfo imageInfo = new ImageInfo();
            try {
                // todo: get images from assets
                InputStreamReader inputReader = new InputStreamReader(context.getAssets().open(image_txt));
                BufferedReader buffReader = new BufferedReader(inputReader);
                String line;
                int line_number = 0;
                float[][][][] x = new float[batch_size][imageInfo.getHeight()][imageInfo.getWidth()][imageInfo.getChannel()];
                int batch_size_iter = 0;
                while ((line = buffReader.readLine()) != null) {
                    System.out.println(path + line);
                    Mat image = TrainerStreamUtils.getImage(path + line, imageInfo);
                    int label = Integer.parseInt(line.split("/")[1]);
                    float[][] label_oneHot = new float[batch_size][imageInfo.getLabel_num()];
                    label_oneHot[batch_size_iter][label] = 1;
                    assert image != null;
                    DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                    if (batch_size_iter < batch_size - 1) {
                        batch_size_iter++;
                        line_number++;
                        System.out.println(line + " " + line_number + " ");
                        continue;
                    } else {
                        batch_size_iter = 0;
                    }
                    Session.Runner runner = session.runner()
                            .feed("x", Tensor.create(x))
                            .feed("y", Tensor.create(label_oneHot));
                    runner.addTarget(optimizerName).run();
                    float[] loss = new float[1];
                    Tensor train_loss = runner.fetch(lossName).run().get(0);
                    train_loss.copyTo(loss);
                    total_loss += loss[0];
                    train_loss_view.setText(line + ": " + line_number + ": " + loss[0]);
                }
                total_loss = total_loss / ((float) line_number / batch_size);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return total_loss;
        }

        public ValueReply computeStream(ComputationGrpc.ComputationBlockingStub stub,
                                        List<Layer> layerList, int layer_size) {
            ValueReply valueReply = null;
            TrainerStreamUtils trainerStreamUtils = new TrainerStreamUtils();
            for (int i = 0; i < layer_size; i++) {
                if (layerList.get(i).getLayerName().equals("non_trainable")) {
                    continue;
                }
                Tensor weights = session.runner().
                        fetch(layerList.get(i).getLayerName()).run().get(0);
                modelTrainableWeighs.put(layerList.get(i).getLayerInitName(), weights);
                modelTrainableInit.put(layerList.get(i).getLayerInitName(), layerList.get(i).getLayerName());
                ClientRequest.Builder clientRequestBuilder = ClientRequest.newBuilder();
                clientRequestBuilder.setToken(token);
                clientRequestBuilder.setId(localId);
                valueReply = trainerStreamUtils.callLayerWeights(clientRequestBuilder,
                        maxFloatNumber, i, stub, weights,
                        layerList.get(i).getLayerTrainableShape());
            }
            return valueReply;
        }
    }
}
