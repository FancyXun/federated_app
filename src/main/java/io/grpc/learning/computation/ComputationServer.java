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

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.learning.api.BaseGraph;
import io.grpc.learning.utils.JsonUtils;
import io.grpc.learning.vo.GraphZoo;
import io.grpc.learning.vo.ModelZooWeights;
import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.TensorVarName;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.tensorflow.*;

import com.alibaba.fastjson.JSON;
import com.google.protobuf.ByteString;

/**
 * Server that manages startup/shutdown of a {@code Computation} server.
 */
public class ComputationServer {
    private static final Logger logger = Logger.getLogger(ComputationServer.class.getName());
    private static final String url = "io.grpc.learning.api";
    private Server server;

    private void start() throws IOException {
        /* The port on which the server should run */
        String localIP;
        Enumeration<NetworkInterface> n = NetworkInterface.getNetworkInterfaces();
        try {
            NetworkInterface e = n.nextElement();
            Enumeration<InetAddress> a = e.getInetAddresses();
            a.nextElement();
            InetAddress addr = a.nextElement();
            localIP = addr.getHostAddress();
        } catch (Exception e1) {
            localIP = "127.0.0.1";
        }

        int port = 50051;
        server = ServerBuilder.forPort(port)
                .addService(new ComputationImpl())
                .build()
                .start();
        logger.info("Server started, ip is " + localIP + " listening on " + port);
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

        @Override
        public void call(ComputationRequest req, StreamObserver<ComputationReply> responseObserver) {
            String clientId = req.getId();
            String node_name = req.getNodeName();
            logger.info("Server received request " + url + "." + node_name + " from " + clientId);
            Graph graph = getGraph(node_name);
            byte[] byteGraph = graph.toGraphDef();
            ComputationReply.Builder reply = ComputationReply.newBuilder();
            reply.setMessage("Received request from " + clientId);
            reply.setGraph(ByteString.copyFrom(byteGraph));
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

        /**
         *
         * @param req
         * @param responseObserver
         */
        @Override
        public void callValue(ComputationRequest req, StreamObserver<TensorValue> responseObserver) {
            String clientId = req.getId();
            String node_name = req.getNodeName();
            int offset = req.getOffset();
            ModelZooWeights modelZooWeights = new ModelZooWeights();
            SequenceData sequenceData = modelZooWeights.getModelZoo().get(node_name);
            GraphZoo graphZoo = new GraphZoo();
            if (sequenceData == null) {
                String s = JsonUtils.readJsonFile(graphZoo.getGraphZooPath().get(node_name)
                        .replace(".pb", ".json"));
                TensorVarName tensorVarName = JsonUtils.jsonToMap(JSON.parseObject(s));
                sequenceData = this.initializerSequence(tensorVarName);
            }
            TensorValue.Builder reply = this.offsetStreamReply(sequenceData, offset);
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

        /**
         *
         * @param request
         * @param responseObserver
         */
        @Override
        public void sendValue(TensorValue request, StreamObserver<ValueReply> responseObserver) {
            super.sendValue(request, responseObserver);
        }

        private Graph getGraph(String node_name) {
            GraphZoo graphZoo = new GraphZoo();
            Graph graph = graphZoo.getGraphZoo().get(node_name);
            if (graph == null) {
                try {
                    ClassLoader classLoader = Class.forName(url + "." + node_name).getClassLoader();
                    BaseGraph basegraph = (BaseGraph) classLoader.loadClass(url + "." + node_name).newInstance();
                    graph = basegraph.getGraph();
                    graphZoo.getGraphZoo().put(node_name, graph);
                    graphZoo.getGraphZooPath().put(node_name, basegraph.pbPath);
                } catch (Exception ClassNotFoundException) {
                    throw new RuntimeException();
                }
            }
            return graph;
        }

        /**
         * Initialize SequenceData from TensorVarName
         * @param tensorVarName
         * @return SequenceData
         */
        private SequenceData initializerSequence(TensorVarName tensorVarName) {
            SequenceData sequenceData = new SequenceData();
            for (int i = 0; i < tensorVarName.getTensorName().size(); i++) {
                List<Integer> integerList = tensorVarName.getTensorShape().get(i);
                String tensorName = tensorVarName.getTensorName().get(i);
                String tensorTargetName = tensorVarName.getTensorTargetName().get(i);
                String tensorAssignName = tensorVarName.getTensorAssignName().get(i);
                sequenceData.getTensorName().add(tensorName);
                sequenceData.getTensorAssignName().add(tensorAssignName);
                sequenceData.getTensorTargetName().add(tensorTargetName);
                sequenceData.getTensorShape().add(integerList);
                int varSize = 0;
                if (integerList.size() == 1) {
                    varSize = integerList.get(0);
                }
                if (integerList.size() == 2) {
                    varSize = integerList.get(0) * integerList.get(1);
                }
                List<Float> var = new ArrayList<>(varSize);
                for (int ii = 0; ii < varSize; ii++) {
                    var.add(0f);
                }
                sequenceData.getTensorVar().add(var);
            }
            for (int i = 0; i < tensorVarName.getPlaceholder().size(); i++) {
                sequenceData.getPlaceholder().add(tensorVarName.getPlaceholder().get(i));
            }
            return sequenceData;
        }

        /**
         *
         * @param sequenceData
         * @param offset
         * @return
         */
        private TensorValue.Builder offsetStreamReply(SequenceData sequenceData, int offset) {
            TensorValue.Builder reply = TensorValue.newBuilder();
            TrainableVarName.Builder trainableVarName = TrainableVarName.newBuilder();
            trainableVarName.setName(sequenceData.getTensorName().get(offset));
            trainableVarName.setTargetName(sequenceData.getTensorTargetName().get(offset));
            reply.setValueSize(sequenceData.getTensorVar().size());
            reply.addAllShapeArray(sequenceData.getTensorShape().get(offset));
            reply.addAllListArray(sequenceData.getTensorVar().get(offset));
            reply.setTrainableName(trainableVarName);
            reply.addAllAssignName(sequenceData.getTensorAssignName());
            reply.addAllPlaceholder(sequenceData.getPlaceholder());
            return reply;
        }
    }
}
