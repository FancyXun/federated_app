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
import java.util.List;
import java.util.UUID;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
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
        private Session session;
        private final int maxFloatNumber = 1000000;
        private String server_ip = "192.168.0.102";
        private int server_port = 50051;
        private final String path = "http://52.81.162.253:8000/res/CASIA-WebFace-aligned"; // image url
        private final String image_txt = "images.txt"; //train images
        private static String localId = UUID.randomUUID().toString().replaceAll("-", "");
        private List<Layer> layerList;
        private int round;


        protected LocalTraining(Activity activity, Context context) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
        }

        @Override
        protected String doInBackground(String... params) {
            runOneRound();
            return "training finished !";
        }

        public Session init(Session session, String initName) {
            session.runner().addTarget(initName).run();
            return session;
        }

        public void runOneRound(){
            channel = ManagedChannelBuilder
                    .forAddress(server_ip, server_port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ClientRequest.Builder builder = ClientRequest.newBuilder().setId(localId);
            Model model = stub.callModel(builder.build());
            Activity activity = activityReference.get();
            getModelGraph(model);
            for (int r = 0; r < round; r++) {
                int layer_size = layerList.size();
                session = init(session, initName);
                if (r != 0) {
                    for (int i = 0; i < layer_size; i++) {
                        Layer layer = layerList.get(i);
                        session.runner().feed(layer.getLayerInitName(),
                                TrainerStreamUtils.getLayerWeights(localId, i, stub))
                                .addTarget(layer.getLayerName() + "/Assign")
                                .run();
                    }
                }
                TextView train_loss_view = null;
                if (activity != null) {
                    train_loss_view = activity.findViewById(R.id.TrainLoss);
                }
                float loss = train(train_loss_view);
                assert train_loss_view != null;
                train_loss_view.setText("round " + r + ": " + loss);
                computeStream(stub, layerList, layer_size);
                stub.computeFinish(builder.build());
            }
        }

        public void getModelGraph(Model model) {
            round = model.getRound();
            round = 10;
            dataSplit = model.getMessage();
            layerList = model.getLayerList();
            List<Meta> metaList = model.getMetaList();
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
                Tensor weights = session.runner().
                        fetch(layerList.get(i).getLayerName()).run().get(0);
                valueReply = trainerStreamUtils.callLayerWeights(maxFloatNumber, i, stub, weights,
                        layerList.get(i).getLayerShape());
            }
            return valueReply;
        }
    }
}
