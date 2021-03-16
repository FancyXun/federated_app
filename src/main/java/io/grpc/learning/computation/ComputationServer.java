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
import io.grpc.learning.model.ModelHelper;
import io.grpc.learning.model.Updater;
import io.grpc.learning.vo.Client;
import io.grpc.stub.StreamObserver;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import com.google.protobuf.ByteString;

import org.jetbrains.bio.npy.NpzFile;
import org.tensorflow.Graph;


/**
 * Server that manages startup/shutdown of a {@code Computation} server.
 */
public class ComputationServer {
    private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(ComputationServer.class.getName());
    private Server server;
    public ModelHelper modelHelper;
    // Determine what logging framework SLF4J is bound to:
//    final StaticLoggerBinder binder = StaticLoggerBinder.getSingleton();

//    static {
//        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
//
//        ctx.getLogger("io.netty").setLevel(Level.INFO);
//    }


    private void start() throws IOException {
        /* initialize the model and graph */
        modelHelper = new ModelHelper();
        modelHelper.loadModel();
        modelHelper.LoadModelWeights();
        ch.qos.logback.classic.Logger logbackLogger =
                (ch.qos.logback.classic.Logger) logger;
        logbackLogger.setLevel(Level.INFO);
//        Logger root = (Logger) LoggerFactory.getLogger(ComputationServer.class.getName());
        logger.info(ModelHelper.getInstance().toString() + " initialization finished");
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
                .addService(new ComputationImpl(modelHelper))
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
        public int minRequestNum = 2;
        public int finished =0;
        public int maxBlock = 4;
        public String token = UUID.randomUUID().toString();
        public String state = "ready";
        public boolean firstRound = true;
        public List<String> AggregationClients = new ArrayList<>();
        public Client client = new Client();
        public ModelHelper modelHelper;
        public int currentRound = 0;

        public ComputationImpl(ModelHelper modelHelper){
            this.modelHelper = modelHelper;
        }

        @Override
        public void callTraining(ClientRequest request, StreamObserver<Certificate> responseObserver) {
            String client_id = request.getId();
            client.getCallTrainingClients().add(client_id);
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
            client.getCallModelClients().add(client_id);
            System.out.println("Receive callModel request from " + client_id );
            Graph graph = modelHelper.getGraph();
            LinkedHashMap<String, String> modelTrainableMap = modelHelper.getModelTrainableMap();
            LinkedHashMap<String, String> modelInitMap = modelHelper.getModelInitMap();
            LinkedHashMap<String, String> metaMap = modelHelper.getMetaMap();
            // set model graph
            Model.Builder model = Model.newBuilder();
            model.setGraph(ByteString.copyFrom(graph.toGraphDef()));
            // set model layer and layer shape
            int layer_index = 0;
            for (String key: modelTrainableMap.keySet()) {
                Layer.Builder layer = Layer.newBuilder();
                layer.setLayerInitName(key);
                layer.setLayerName(modelTrainableMap.get(key));
//                layer.setLayerTrainableShape(modelInitMap.get(key));
                layer.setLayerShape(modelInitMap.get(key));
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
            client.getCallLayerWeightsClients().add(client_id);
            System.out.println("Receive callLayerWeights request from " + client_id);
            String layer_name = request.getLayerName();
            responseObserver.onNext(modelHelper.getLayerWeightsHashMap().get(layer_name).build());
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
            client.getComputeLayerWeightsClients().add(request.getId());
            if (client_token.equals(token) && state.equals("ready")){
                // valid token
                File file=new File(modelHelper.getModelWeighsPath()+"/"+clientRequest.getId());
                if (!file.exists()) {
                    file.mkdir();
                }
                Path filePath = new File(modelHelper.getModelWeighsPath()+"/"+clientRequest.getId()+"/"+
                        request.getLayerName().replace("/","_")
                        +"__"+(int)request.getPart()+".npz").toPath();
                NpzFile.Writer writer = NpzFile.write(filePath, true);

                float[] arr = new float[request.getTensor().getFloatValList().size()];
                int index = 0;
                for (Float value: request.getTensor().getFloatValList()) {
                    arr[index++] = value;
                }
                writer.write("layer_entry", arr);
                writer.close();
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
        public void computeMetrics(TrainMetrics request, StreamObserver<ValueReply> responseObserver) {
            ValueReply.Builder valueReplyBuilder = ValueReply.newBuilder();
            List<Float> accFloatList = request.getAccValueList();
            List<Float> lossFloatList = request.getLossValueList();
            List<Float> accEvalFloatList = request.getEvalAccValueList();
            List<Float> lossEvalFloatList = request.getEvalLossValueList();
            String client_id = request.getId();
            int round = request.getRound();
            File file=new File(modelHelper.getModelWeighsPath()+"/"+client_id+"/"+round);
            if (!file.exists()) {
                file.mkdir();
            }

            FileWriter writer = null;
            try {
                writer = new FileWriter(modelHelper.getModelWeighsPath()+"/"+client_id+"/"+round + "/accFloat.txt");
                for(Float acc: accFloatList) {
                    writer.write(acc + System.lineSeparator());
                }
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            try {
                writer = new FileWriter(modelHelper.getModelWeighsPath()+"/"+client_id+"/"+round + "/lossFloat.txt");
                for(Float loss: lossFloatList) {
                    writer.write(loss + System.lineSeparator());
                }
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            try {
                writer = new FileWriter(modelHelper.getModelWeighsPath()+"/"+client_id+"/"+round + "/accEvalFloat.txt");
                for(Float accEval: accEvalFloatList) {
                    writer.write(accEval + System.lineSeparator());
                }
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            try {
                writer = new FileWriter(modelHelper.getModelWeighsPath()+"/"+client_id+"/"+round + "/lossEvalFloat.txt");
                for(Float lossEval: lossEvalFloatList) {
                    writer.write(lossEval + System.lineSeparator());
                }
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            responseObserver.onNext(valueReplyBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void computeFinish(ClientRequest request, StreamObserver<ValueReply> responseObserver) {
            ValueReply.Builder valueReplyBuilder = ValueReply.newBuilder();
            valueReplyBuilder.setMessage(true);
            if (state.equals("ready")) {
                AggregationClients.add(request.getId());
            }
            client.getComputeFinishClients().add(request.getId());
            System.out.println(AggregationClients.stream()
                    .map(Object::toString)
                    .collect(Collectors.joining("\n")));
            synchronized(this){
                if (AggregationClients.size() >= minRequestNum){
                    state = "wait";
                    File f = new File(modelHelper.getModelWeighsPath()+"/" +"aggClients.txt");
                    try{
                        BufferedWriter bw = new BufferedWriter(new FileWriter(f, false));
                        bw.write(AggregationClients.stream()
                                .map(Object::toString)
                                .collect(Collectors.joining("\n")));
                        bw.close();
                    }catch(IOException e){
                        e.printStackTrace();
                    }

                    modelHelper.updateWeights();
                    AggregationClients.clear();
                    modelHelper.LoadModelWeights();
                    firstRound = false;
                    token = UUID.randomUUID().toString();
                    state = "ready";
                }
            }
            
            responseObserver.onNext(valueReplyBuilder.build());
            responseObserver.onCompleted();
        }
    }
}
