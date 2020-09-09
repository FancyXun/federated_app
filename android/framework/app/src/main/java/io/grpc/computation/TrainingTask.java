package io.grpc.computation;

import android.app.Activity;
import android.os.AsyncTask;
import android.text.TextUtils;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.WeakReference;
import java.util.Iterator;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationReply;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.utils.LocalCSVReader;

public class TrainingTask {
     static class LocalTrainingTask extends AsyncTask<String, Void, String> {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;

        protected LocalTrainingTask(Activity activity) {
            this.activityReference = new WeakReference<Activity>(activity);
        }
        @Override
        protected String doInBackground(String... params) {
            String host = params[0];
            String portStr = params[1];
            String local_id = UUID.randomUUID().toString().replaceAll("-", "");
            String dataPath = params[2];
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.valueOf(portStr);
            try {
                channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
                ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
                ComputationRequest request = ComputationRequest.newBuilder().setId(local_id).setNodeName("LogisticsRegression").build();
                ComputationReply reply = stub.call(request);
                Graph graph = new Graph();
                // todo: implement bp in android device
                graph.importGraphDef(reply.getGraph().toByteArray());
                LocalCSVReader localCSVReader = new LocalCSVReader(dataPath, 0,"target");
                float [][] x = localCSVReader.getX();
                float [][] y = localCSVReader.getY_oneHot();
                float [] b = new float[localCSVReader.getY_oneHot()[0].length];
                float [][] w = new float[localCSVReader.getX()[0].length][localCSVReader.getY_oneHot()[0].length];
                Session session = new Session(graph);
                session.runner().feed("w/init", Tensor.create(w)).addTarget("w/Assign").run();
                session.runner().feed("b/init", Tensor.create(b)).addTarget("b/Assign").run();
                Tensor tensor = null;
                for (int i = 0; i < 10; i++){
                    tensor = session.runner().fetch("cost").feed("x", Tensor.create(x))
                            .feed("y", Tensor.create(y)).run().get(0);
                    session.runner().feed("x", Tensor.create(x))
                            .feed("y", Tensor.create(y)).addTarget("minimizeGradientDescent").run();
                    System.out.println(tensor.floatValue());
                }
                if (null == tensor){
                    return "";
                }
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
             Button train_button = (Button) activity.findViewById(R.id.train_button);
             resultText.setText(result);
             train_button.setEnabled(true);
         }
    }
}
