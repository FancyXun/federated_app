package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.Graph;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.WeakReference;
import java.util.List;
import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.api.SessionMetrics;
import io.grpc.api.SessionRunner;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationReply;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.utils.LocalCSVReader;
import io.grpc.utils.StateInfo;
import io.grpc.vo.Metrics;
import io.grpc.vo.SequenceType;

public class TrainingTask {
    static class LocalTrainingTask extends StreamCallTask {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;
        @SuppressLint("StaticFieldLeak")
        private Context context;
        @SuppressLint("StaticFieldLeak")
        private TextView textView;
        private StateInfo stateInfo;
        // server IP and port
        private String host = "192.168.50.13";
        private int port = 50051;
        // the round of federated training
        private int round = 0;
        private String dataSplit = "train@0-8";


        protected LocalTrainingTask(Activity activity, Context context, TextView textView) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
            this.textView = textView;
            this.stateInfo = new StateInfo();
        }

        @SuppressLint("WrongThread")
        @Override
        protected String doInBackground(String... params) {
            String action = params[4];
            if (action.equals("training")){
                float loss;
                try {
                    loss = this.runOneRound(params);;
                    while (round > 0) {
                        params[3] = String.valueOf(round);
                        loss = this.runOneRound(params);
                    }
                    return "Loss is: " + loss;
                } catch (Exception e) {
                    StringWriter sw = new StringWriter();
                    PrintWriter pw = new PrintWriter(sw);
                    e.printStackTrace(pw);
                    pw.flush();
                    return String.format("Failed... : %n%s", sw);
                }
            }
            else{
                return this.inference(params);
            }

        }

        @Override
        protected void onPostExecute(String result) {
            if (channel != null){
                try {
                    channel.shutdown().awaitTermination(1, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            Activity activity = activityReference.get();
            if (activity == null) {
                return;
            }
            TextView resultText = (TextView) activity.findViewById(R.id.server_response_text);
            Button train_button = (Button) activity.findViewById(R.id.train_button);
            resultText.setText(result);
            train_button.setEnabled(true);
        }

        protected ComputationReply callRound(String... params) {
            String localId = params[0];
            String modelName = params[2];
            channel = ManagedChannelBuilder
                    .forAddress(this.host, this.port)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ComputationRequest.Builder builder = ComputationRequest.newBuilder().setId(localId)
                    .setNodeName(modelName);
            ComputationReply reply = stub.call(builder.build());
            return reply;
        }

        /**
         * Orchestration logic for one round of optimization.
         * @param params
         * @return
         */
        protected float runOneRound(String... params) {
            String localId = params[0];
            String dataPath = params[1];
            String modelName = params[2];
            channel = ManagedChannelBuilder
                    .forAddress(this.host, this.port)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ComputationRequest.Builder builder = ComputationRequest.newBuilder().setId(localId)
                    .setNodeName(modelName).setAction("training");
            ComputationReply reply = stub.call(builder.build());
            round = reply.getRound();
            dataSplit = reply.getMessage();
            Graph graph = new Graph();
            // Get graph from server
            // todo: implement bp in android device,
            graph.importGraphDef(reply.getGraph().toByteArray());
            // Get model weights from server
            SequenceType sequenceType = this.SequenceCall(stub, builder);
            // Load data
            LocalCSVReader localCSVReader = new LocalCSVReader(
                    this.context, dataPath, 0, "target", dataSplit, "training");
            SessionRunner runner = new SessionRunner(this.context, graph, sequenceType,
                    localCSVReader, round);
            List<List<Float>> tensorVar = runner.invoke(this.textView);
            runner.eval(this.textView);
            // Set metrics
            Metrics metrics = this.setMetrics(runner);
            metrics.weights = localCSVReader.getHeight();
            this.upload(stub, localId, modelName, tensorVar, metrics);
            this.stateInfo.setStateCode(1);
            return runner.metricsEntity.getLoss();
        }

        protected String inference(String... params){
            String localId = params[0];
            String dataPath = params[1];
            String modelName = params[2];
            channel = ManagedChannelBuilder
                    .forAddress(this.host, this.port)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            ComputationRequest.Builder builder = ComputationRequest.newBuilder().setId(localId)
                    .setNodeName(modelName).setAction("inference");
            ComputationReply reply = stub.call(builder.build());
            dataSplit = reply.getMessage();
            Graph graph = new Graph();
            // Get graph from server
            graph.importGraphDef(reply.getGraph().toByteArray());
            // Get model weights from server
            SequenceType sequenceType = this.SequenceCall(stub, builder);
            // Load data
            LocalCSVReader localCSVReader = new LocalCSVReader(
                    this.context, dataPath, 0, "target", dataSplit, "inference");
            SessionRunner runner = new SessionRunner(this.context, graph, sequenceType,
                    localCSVReader, round);
            return runner.inference(this.textView);
        }

        public Metrics setMetrics(SessionRunner runner){
            Metrics metrics = new Metrics();
            metrics.metricsName.add("train_loss");
            metrics.metrics.add(runner.metricsEntity.getLoss());
            metrics.metricsName.add("eval_loss");
            metrics.metrics.add(runner.metricsEntity.getEval_loss());
            metrics.metricsName.add("train_auc");
            metrics.metrics.add(runner.metricsEntity.getAUC());
            metrics.metricsName.add("eval_auc");
            metrics.metrics.add(runner.metricsEntity.getEval_AUC());
            return metrics;
        }

    }
}
