/*
 * Copyright 2015 The gRPC Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.grpc.computation;

import android.app.Activity;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationReply;
import io.grpc.learning.computation.ComputationRequest;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;
import java.util.UUID;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class ComputationActivity extends AppCompatActivity {
    private Button sendButton;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_computation);
        sendButton = (Button) findViewById(R.id.send_button);
        resultText = (TextView) findViewById(R.id.server_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
    }

    public void sendId(View view) {
        sendButton.setEnabled(false);
        resultText.setText("");
        new ComputationTask(this)
                .execute(
                        "localhost",
                        "8080");
    }

    private static class ComputationTask extends AsyncTask<String, Void, String> {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;

        private ComputationTask(Activity activity) {
            this.activityReference = new WeakReference<Activity>(activity);
        }

        @Override
        protected String doInBackground(String... params) {
            String host = params[0];
            String portStr = params[1];
            String local_id = UUID.randomUUID().toString().replaceAll("-","");
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.valueOf(portStr);
            try {
                channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
                ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
                ComputationRequest request = ComputationRequest.newBuilder().setId(local_id).build();
                ComputationReply reply = stub.call(request);
                reply.getGraph().toByteArray();
                Graph graph = new Graph();
//                graph.importGraphDef(reply.getGraph().toByteArray());
//                Session session = new Session(graph);
//                Tensor tensor = session.runner().fetch("xy").feed("x", Tensor.create(5.0f)).feed("y", Tensor.create(2.0f)).run().get(0);
                return reply.getMessage();
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
            Button sendButton = (Button) activity.findViewById(R.id.send_button);
            resultText.setText(result);
            sendButton.setEnabled(true);
        }
    }
}
