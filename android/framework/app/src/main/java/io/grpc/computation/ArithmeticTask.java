package io.grpc.computation;

import android.app.Activity;
import android.os.AsyncTask;
import android.text.TextUtils;
import android.widget.TextView;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.WeakReference;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationReply;
import io.grpc.learning.computation.ComputationRequest;

public class ArithmeticTask {
    static class MulTask extends AsyncTask<String, Void, String> {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;

        MulTask(Activity activity) {
            this.activityReference = new WeakReference<Activity>(activity);
        }

        @Override
        protected String doInBackground(String... params) {
            String host = params[0];
            String portStr = params[1];
            String xStr = params[2];
            String yStr = params[3];
            float x;
            float y;
            try {
                x = Float.parseFloat(xStr);
                y = Float.parseFloat(yStr);
            } catch (Exception e) {
                return "Failed to convert string to float";
            }

            String local_id = UUID.randomUUID().toString().replaceAll("-", "");
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.parseInt(portStr);
            try {
                channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
                ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
                ComputationRequest request = ComputationRequest.newBuilder().setId(local_id).setNodeName("FloatMul").build();
                ComputationReply reply = stub.call(request);
                Graph graph = new Graph();
                graph.importGraphDef(reply.getGraph().toByteArray());
                Session session = new Session(graph);
                Tensor tensor = session.runner().fetch("xy").feed("x", Tensor.create(x)).feed("y", Tensor.create(y)).run().get(0);
                return String.valueOf(tensor.floatValue());
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
            resultText.setText(result);
        }
    }
}
