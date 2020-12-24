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

import ch.qos.logback.classic.Level;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.learning.logging.SystemOut;
import io.grpc.learning.model.Initializer;
import io.grpc.learning.model.Updater;
import io.grpc.stub.StreamObserver;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import com.google.protobuf.ByteString;

import org.apache.logging.log4j.LogManager;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.slf4j.impl.StaticLoggerBinder;

import javax.security.auth.login.Configuration;


/**
 * Server that manages startup/shutdown of a {@code Computation} server.
 */
public class ComputationServer {
    private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(ComputationServer.class.getName());
    private Server server;
    public Initializer initializer;
    // Determine what logging framework SLF4J is bound to:
//    final StaticLoggerBinder binder = StaticLoggerBinder.getSingleton();

//    static {
//        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
//
//        ctx.getLogger("io.netty").setLevel(Level.INFO);
//    }


    private void start() throws IOException {
        /* initialize the model and graph */
        ch.qos.logback.classic.Logger logbackLogger =
                (ch.qos.logback.classic.Logger) logger;
        logbackLogger.setLevel(Level.INFO);
//        Logger root = (Logger) LoggerFactory.getLogger(ComputationServer.class.getName());
        Initializer.getInstance().loadModel(1);
        logger.info(Initializer.getInstance().toString() + " initialization finished");
        // this will print the name of the logger factory to stdout
//        System.out.println(binder.getLoggerFactoryClassStr());
        /* The port on which the server should run */
        String localIP;
        Enumeration<NetworkInterface> n = NetworkInterface.getNetworkInterfaces();
        try {
            NetworkInterface e = n.nextElement();
            Enumeration<InetAddress> a = e.getInetAddresses();
            a.nextElement();
            InetAddress address = a.nextElement();
            localIP = address.getHostAddress();
        } catch (Exception e1) {
            localIP = "127.0.0.1";
        }
        int port = 50051;
        logger.info(ComputationServer.class.getName() +": " +localIP + ":" + port);
        server = ServerBuilder.forPort(port)
                .addService(new ComputationImpl())
                .build()
                .start();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    ComputationServer.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }

    private void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    /**
     * Main launches the server from the command line.
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        final ComputationServer server = new ComputationServer();
        server.start();
        server.blockUntilShutdown();
    }

    static class ComputationImpl extends ComputationGrpc.ComputationImplBase {
        public int minRequestNum = 1;
        public int finished =0;
        public int maxBlock = 4;
        public HashMap<String, Integer> currentBlock = new HashMap<>();
        public String token = UUID.randomUUID().toString();
        public String state = "ready";
        public boolean firstRound = true;
        public List<String> FinishedClient = new ArrayList<>();

        @Override
        public void callTraining(ClientRequest request, StreamObserver<Certificate> responseObserver) {
            String client_id = request.getId();
            System.out.println("Receive callTraining request from " + client_id );
            Certificate.Builder cBuilder = Certificate.newBuilder();
            cBuilder.setServerState(state);
            cBuilder.setToken(token);
            responseObserver.onNext(cBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void callModel(ClientRequest request, StreamObserver<Model> responseObserver) {
            String client_id = request.getId();
            if (currentBlock.containsKey(client_id)){
                if (currentBlock.get(client_id) > maxBlock){
                    currentBlock.put(client_id, 1);
                }
                else{
                    currentBlock.put(client_id, currentBlock.get(client_id) + 1);
                }
            }
            else{
                currentBlock.put(client_id, 1);
            }
            System.out.println("Receive callModel request from " + client_id + ": "+ currentBlock.get(client_id));
            Initializer.getInstance().loadModel(currentBlock.get(client_id));
            Initializer initializer = Initializer.getInstance();
            Graph graph = initializer.getGraph();
            LinkedHashMap<String, String> modelTrainableMap = initializer.getModelTrainableMap();
            LinkedHashMap<String, String> modelInitMap = initializer.getModelInitMap();
            LinkedHashMap<String, String> metaMap = initializer.getMetaMap();
            // set model graph
            Model.Builder model = Model.newBuilder();
            model.setGraph(ByteString.copyFrom(graph.toGraphDef()));
            // set model layer and layer shape
            int layer_index = 0;
            String [][] strings = new String[modelInitMap.size()][2];
            int i =0;
            for (String key : modelTrainableMap.keySet()) {
                strings[i][0] = key;
                i++;
            }
            int j =0;
            for (String key : modelInitMap.keySet()) {
                strings[j][1] = key;
                j++;
            }
            for (i =0; i< modelInitMap.size(); i++) {
                Layer.Builder layer = Layer.newBuilder();
                if (strings[i][0] == null){
                    layer.setLayerName("non_trainable");
                }
                else{
                    layer.setLayerName(strings[i][0]);
                    layer.setLayerTrainableShape(modelTrainableMap.get(strings[i][0]));
                }
                layer.setLayerShape(modelInitMap.get(strings[i][1]));
                layer.setLayerInitName(strings[i][1]);
                model.addLayer(layer_index, layer);
                layer_index++;
            }
            // set model meta and its shape if necessary
            int meta_index = 0;
            for (String key : metaMap.keySet()) {
                Meta.Builder meta = Meta.newBuilder();
                meta.setMetaName(key);
                meta.setMetaShape(metaMap.get(key));
                model.addMeta(meta_index, meta);
                meta_index++;
            }
            model.setFirstRound(firstRound);
            responseObserver.onNext(model.build());
            responseObserver.onCompleted();
        }

        @Override
        public void callLayerWeights(LayerWeightsRequest request, StreamObserver<LayerWeights> responseObserver) {
            String client_id = request.getId();
            System.out.println("Receive callLayerWeights request from " + client_id);
            Updater updater = Updater.getInstance();
            int layer_id = (int) request.getLayerId();
            responseObserver.onNext(updater.layerWeightsArrayList.get(layer_id).build());
            responseObserver.onCompleted();
        }

        // todo: Received DATA frame for an unknown stream error
        //  https://github.com/grpc/grpc-java/issues/4651
        @Deprecated
        @Override
        public void callModelWeights(ClientRequest request,
                                     StreamObserver<io.grpc.learning.computation.ModelWeights> responseObserver) {
            String client_id = request.getId();
            System.out.println("Receive callModelWeights request from " + client_id);
            Updater updater = Updater.getInstance();
            responseObserver.onNext(updater.modelWeightsBuilder.build());
            responseObserver.onCompleted();
        }

        // todo: Received DATA frame for an unknown stream error
        //  https://github.com/grpc/grpc-java/issues/4651
        @Deprecated
        @Override
        public void computeWeights(io.grpc.learning.computation.ModelWeights request, StreamObserver<ValueReply> responseObserver) {
            ValueReply.Builder valueReplyBuilder = ValueReply.newBuilder();
            valueReplyBuilder.setMessage(true);
            responseObserver.onNext(valueReplyBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void computeLayerWeights(LayerWeights request, StreamObserver<ValueReply> responseObserver) {
            ClientRequest clientRequest = request.getClientRequest();
            String client_token = clientRequest.getToken();
            if (client_token.equals(token) && state.equals("ready")){
                // valid token
                File file=new File("/tmp/model_weights/"+clientRequest.getId());
                if (!file.exists()) {
                    file.mkdir();
                }
                File f = new File("/tmp/model_weights/"+clientRequest.getId()+"/layer_" +request.getLayerId()+".txt");
                if (!f.exists()) {
                    try {
                        f.createNewFile();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                try{
                    BufferedWriter bw = new BufferedWriter(new FileWriter(f, true));
                    bw.write(String.valueOf(request.getTensor().getFloatValList()));
                    bw.close();
                }catch(IOException e){
                    e.printStackTrace();
                }
            }
            ValueReply.Builder valueReplyBuilder = ValueReply.newBuilder();
            Certificate.Builder cBuilder =  Certificate.newBuilder();
            cBuilder.setToken(token);
            cBuilder.setServerState(state);
            valueReplyBuilder.setCertificate(cBuilder);
            responseObserver.onNext(valueReplyBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void computeFinish(ClientRequest request, StreamObserver<ValueReply> responseObserver) {
            ValueReply.Builder valueReplyBuilder = ValueReply.newBuilder();
            valueReplyBuilder.setMessage(true);
            FinishedClient.add(request.getId());
            System.out.println(FinishedClient.toArray().toString());
            responseObserver.onNext(valueReplyBuilder.build());
            responseObserver.onCompleted();
        }
    }
}
