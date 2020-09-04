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

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private Button sendButton;
    private TextView resultText;
    private EditText hostEdit;
    private EditText portEdit;
    private EditText x;
    private EditText y;
    private static final String DATA_FILE = "file:///android_asset/";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_computation);
        sendButton = (Button) findViewById(R.id.send_button);
        hostEdit = (EditText) findViewById(R.id.host_edit_text);
        portEdit = (EditText) findViewById(R.id.port_edit_text);
        x = (EditText) findViewById(R.id.x);
        y = (EditText) findViewById(R.id.y);
        resultText = (TextView) findViewById(R.id.server_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
    }

    public void Mul(View view) {
        sendButton.setEnabled(false);
        resultText.setText("");
        new MulTask(this)
                .execute(
                        hostEdit.getText().toString(),
                        portEdit.getText().toString(),
                        x.getText().toString(),
                        y.getText().toString()
                );
    }

    private static class MulTask extends AsyncTask<String, Void, String> {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;

        private MulTask(Activity activity) {
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
                return String.format("Failed to convert string to float");
            }

            String local_id = UUID.randomUUID().toString().replaceAll("-", "");
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.valueOf(portStr);
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
            Button sendButton = (Button) activity.findViewById(R.id.send_button);
            resultText.setText(result);
            sendButton.setEnabled(true);
        }
    }
}
