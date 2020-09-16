package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.os.AsyncTask;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.Graph;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.api.SessionRunner;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationReply;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.learning.computation.TensorValue;
import io.grpc.utils.LocalCSVReader;
import io.grpc.utils.StateInfo;
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
        private String host = "52.81.112.107";
        private int port = 50051;

        protected LocalTrainingTask(Activity activity, Context context, TextView textView) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
            this.textView = textView;
            this.stateInfo = new StateInfo();
        }

        @SuppressLint("WrongThread")
        @Override
        protected String doInBackground(String... params) {
            String local_id = params[0];
            String dataPath = params[1];
            int epoch = Integer.parseInt(params[2]);
            try {
                channel = ManagedChannelBuilder
                        .forAddress(this.host, this.port)
                        .usePlaintext().build();
                ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
                ComputationRequest.Builder builder = ComputationRequest.newBuilder().setId(local_id)
                        .setNodeName("LogisticsRegression");
                ComputationReply reply = stub.call( builder.build());
                Graph graph = new Graph();
                // Get graph from server
                // todo: implement bp in android device,
                graph.importGraphDef(reply.getGraph().toByteArray());
                // Get model weights from server
                SequenceType sequenceType = this.SequenceCall(stub, builder);
                // Load data
                LocalCSVReader localCSVReader = new LocalCSVReader(
                        this.context, dataPath, 0, "target");
                new SessionRunner(graph, sequenceType, localCSVReader, epoch)
                        .invoke(this.textView);
                this.stateInfo.setStateCode(1);
                return "Training finished";
            } catch (Exception e) {
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                e.printStackTrace(pw);
                pw.flush();
                return String.format("Failed... : %n%s", sw);
            }
        }

        @Override
        protected void onPostExecute(String result) {
            try {
                channel.shutdown().awaitTermination(1, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
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
    }
}
