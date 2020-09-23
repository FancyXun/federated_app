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
import io.grpc.learning.utils.JsonUtils;
import io.grpc.learning.vo.GraphZoo;
import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.TaskState;
import io.grpc.learning.vo.TaskZoo;
import io.grpc.learning.vo.TensorVarName;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import com.alibaba.fastjson.JSON;
import com.google.protobuf.ByteString;

import static io.grpc.learning.computation.FederatedComp.getGraph;
import static io.grpc.learning.computation.SequenceComp.initializerSequence;
import static io.grpc.learning.computation.SequenceComp.offsetStreamReply;

/**
 * Server that manages startup/shutdown of a {@code Computation} server.
 */
public class ComputationServer {
    private static final Logger logger = Logger.getLogger(ComputationServer.class.getName());
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
        private Map<String, Integer> requestNum = new HashMap<String, Integer>();
        private int minRequestNum = 3;
        private Map<String, Integer> minRequestMap = new HashMap<String, Integer>();


        @Override
        public void call(ComputationRequest req, StreamObserver<ComputationReply> responseObserver) {
            String nodeName = req.getNodeName();
            if (!minRequestMap.containsKey(nodeName)) {
                minRequestMap.put(nodeName, minRequestNum);
                requestNum.put(nodeName, 0);
            }
            logger.info("Server received request " + nodeName + " from " + req.getId());
            byte[] byteGraph = getGraph(nodeName).toGraphDef();
            ComputationReply.Builder reply = ComputationReply.newBuilder();
            reply.setGraph(ByteString.copyFrom(byteGraph));
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

        /**
         * @param req
         * @param responseObserver
         */
        @Override
        public void callValue(ComputationRequest req, StreamObserver<TensorValue> responseObserver) {
            String nodeName = req.getNodeName();
            int offset = req.getOffset();
            HashMap<String, SequenceData> sequenceDataHashMap = TaskZoo
                    .getTaskQueue().get(req.getId());
            SequenceData sequenceData;
            if (sequenceDataHashMap == null) {
                String s = JsonUtils.readJsonFile(GraphZoo.getGraphZooPath().get(nodeName)
                        .replace(".pb", ".json"));
                TensorVarName tensorVarName = JsonUtils.jsonToMap(JSON.parseObject(s));
                sequenceData = initializerSequence(tensorVarName);
                //  initialize weights for one client
                sequenceDataHashMap = new HashMap<>();
                sequenceDataHashMap.put(nodeName, sequenceData);
                TaskZoo.getTaskQueue().put(req.getId(), sequenceDataHashMap);
                TaskZoo.getTask().put(req.getId() + nodeName, TaskState.callValue);
            } else {
                if (TaskState.sendValue == TaskZoo.getTask().get(req.getId() + nodeName)) {
                    while (requestNum.get(nodeName) < minRequestNum) {
                        System.out.println(requestNum);
                        try {
                            Thread.sleep(10000);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    sequenceData = FederatedComp.aggregation(nodeName);
                    assert sequenceData != null;
                    TaskZoo.getTask().put(req.getId() + nodeName, TaskState.callValue);
                } else {
                    sequenceData = sequenceDataHashMap.get(nodeName);
                }
            }
            TensorValue.Builder reply = offsetStreamReply(sequenceData, offset);
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

        /**
         * @param req
         * @param responseObserver
         */
        @Override
        public void sendValue(TensorValue req, StreamObserver<ValueReply> responseObserver) {
            String nodeName = req.getNodeName();
            SequenceData sequenceData = TaskZoo.getTaskQueue().get(req.getId()).get(nodeName);
            sequenceData.getTensorVar().set(req.getOffset(), req.getListArrayList());
            TaskZoo.getTaskQueue().get(req.getId()).put(nodeName, sequenceData);
            if (TaskZoo.getTaskInt().get(req.getId()) == null) {
                TaskZoo.getTaskInt().put(req.getId(), 1);
            } else {
                TaskZoo.getTaskInt().put(req.getId(),
                        TaskZoo.getTaskInt().get(req.getId()) + 1);
            }
            if (TaskZoo.getTaskInt().get(req.getId()) == req.getValueSize()) {
                requestNum.put(nodeName, requestNum.get(nodeName) + 1);
            }
            ValueReply.Builder reply = ValueReply.newBuilder().setMessage(true);
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
            TaskZoo.getTask().put(req.getId() + nodeName, TaskState.sendValue);
        }
    }
}
