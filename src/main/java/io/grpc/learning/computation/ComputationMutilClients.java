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

package io.grpc.learning.computation;

import computation.TensorEntity;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * A simple client that requests a computation from the {@link ComputationServer}.
 */
public class ComputationMutilClients {
    public static void main(String[] args) throws Exception {
        int clients = 1000;
        ArrayList<Thread> threadArrayList = new ArrayList<>();
        for (int i=0 ; i< clients; i++){
            threadArrayList.add(new Thread(new MyRunnable()));
        }
        for (int i =0 ; i< clients; i++){
            threadArrayList.get(i).start();
        }
    }
}

class MyRunnable implements Runnable {
    private static final Logger logger = Logger.getLogger(ComputationClient.class.getName());

    public LinkedHashMap<String, String> loadModelMeta(String filePath) {

        LinkedHashMap<String, String> map = new LinkedHashMap<String, String>();

        String line;
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(filePath));
            int emptyLine = 0;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(";", 2);
                if (parts.length >= 2) {
                    String key = parts[0];
                    String value = parts[1];
                    map.put(key, value);
                } else {
                    map.put(String.valueOf(emptyLine), "null");
                    emptyLine += 1;
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }

    public static void getLayerWeightsByName(String localId, String layer_name,
                                             ComputationGrpc.ComputationBlockingStub stub) {
        LayerWeightsRequest.Builder layerBuilder = LayerWeightsRequest.newBuilder();
        layerBuilder.setId(localId);
        layerBuilder.setLayerName(layer_name);
        stub.callLayerWeights(layerBuilder.build());
    }

    public void callWeights(String id, ComputationGrpc.ComputationBlockingStub blockingStub) {
        ComputationRequest request = ComputationRequest.newBuilder()
                .setId(id)
                .setNodeName("FloatMul")
                .build();
        ComputationReply response;
        try {
            LinkedHashMap<String, String> modelTrainableMap =
                    loadModelMeta(
                            "/Users/voyager/code/android-demo/federated_app/src/main/python" +
                                    "/face_rec/graph/mobileFaceNet/mobileFaceNet_trainable_var.txt");
            for (String value: modelTrainableMap.values()){
                getLayerWeightsByName(id, value, blockingStub);
            }
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }
    }

    public ValueReply callLayerWeights(ClientRequest.Builder clientRequestBuilder,
                                       String layer_name, ComputationGrpc.ComputationBlockingStub stub,
                                        String layerShape) {
        Pattern p = Pattern.compile("\\d+");
        int maxFloatNumber = 1000000;
        ValueReply valueReply = null;
        LayerWeights.Builder layerWeightsBuilder = LayerWeights.newBuilder();
        TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                TensorEntity.TensorShapeProto.newBuilder();
        Matcher m = p.matcher(layerShape);
        int dim_index = 0;
        int size = 1;
        while (m.find()) {
            int dim = Integer.parseInt(m.group());
            size = size * dim;
            TensorEntity.TensorShapeProto.Dim.Builder dimBuilder =
                    TensorEntity.TensorShapeProto.Dim.newBuilder();
            dimBuilder.setSize(dim);
            tensorShapeBuilder.addDim(dim_index, dimBuilder);
            dim_index++;
        }
        float[] floats = new float[size];
        if (size > maxFloatNumber) {
            int j = 0;
            boolean flag = true;
            TensorEntity.TensorProto.Builder tensorBuilder = null;
            int part = 0;
            while (j < size) {
                if (j == 0) {
                    tensorBuilder =
                            TensorEntity.TensorProto.newBuilder();
                }
                tensorBuilder.addFloatVal(floats[j]);
                if (j == maxFloatNumber - 1) {
                    tensorBuilder.setTensorShape(tensorShapeBuilder);
                    layerWeightsBuilder.setTensor(tensorBuilder);
                    layerWeightsBuilder.setLayerName(layer_name);
                    layerWeightsBuilder.setPart(part);
                    layerWeightsBuilder.setClientRequest(clientRequestBuilder.build());
                    valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
                    j = 0;
                    size = size - maxFloatNumber;
                    part++;
                    if (size == 0) {
                        flag = false;
                    }
                    tensorBuilder.clear();
                } else {
                    j++;
                }
            }
            if (flag) {
                tensorBuilder.setTensorShape(tensorShapeBuilder);
                layerWeightsBuilder.setTensor(tensorBuilder);
                layerWeightsBuilder.setLayerName(layer_name);
                layerWeightsBuilder.setPart(part);
                layerWeightsBuilder.setClientRequest(clientRequestBuilder.build());
                valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
            }
        } else {
            TensorEntity.TensorProto.Builder tensorBuilder =
                    TensorEntity.TensorProto.newBuilder();
            for (int j = 0; j < floats.length; j++) {
                tensorBuilder.addFloatVal(floats[j]);
            }
            tensorBuilder.setTensorShape(tensorShapeBuilder);
            layerWeightsBuilder.setTensor(tensorBuilder);
            layerWeightsBuilder.setLayerName(layer_name);
            layerWeightsBuilder.setPart(0);
            layerWeightsBuilder.setClientRequest(clientRequestBuilder.build());
            valueReply = stub.computeLayerWeights(layerWeightsBuilder.build());
        }
        return valueReply;
    }

    public void streamUpload(ComputationGrpc.ComputationBlockingStub stub,
                             String id) {
        ValueReply valueReply = null;
        LinkedHashMap<String, String> modelTrainableMap =
                loadModelMeta(
                        "/Users/voyager/code/android-demo/federated_app/src/main/python" +
                                "/face_rec/graph/mobileFaceNet/mobileFaceNet_trainable_var.txt");
        LinkedHashMap<String, String> modelInitMap =
                loadModelMeta(
                        "/Users/voyager/code/android-demo/federated_app/src/main/python" +
                                "/face_rec/graph/mobileFaceNet/mobileFaceNet_global_var.txt");
        for (String key: modelTrainableMap.keySet()) {
            System.out.println(modelInitMap.get(key));
            ClientRequest.Builder clientRequestBuilder = ClientRequest.newBuilder();
            clientRequestBuilder.setId(id);
            valueReply = callLayerWeights(clientRequestBuilder,
                    key, stub,
                    modelInitMap.get(key));
        }
    }

    public void run() {
        String local_id = UUID.randomUUID().toString().replaceAll("-","");
        // Access a service running on the local machine on port 50051
        String target = "localhost:50051";
        // Create a communication channel to the server, known as a Channel. Channels are thread-safe
        // and reusable. It is common to create channels at the beginning of your application and reuse
        // them until the application shuts down.
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext()
                .build();
        ComputationGrpc.ComputationBlockingStub blockingStub = ComputationGrpc.newBlockingStub(channel);

        try {
//            callWeights(local_id,blockingStub);
            streamUpload(blockingStub, local_id);
        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            try {
                channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

