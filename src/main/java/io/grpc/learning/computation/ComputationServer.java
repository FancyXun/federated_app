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
import io.grpc.learning.vo.GraphZoo;
import io.grpc.learning.vo.ModelWeights;
import io.grpc.learning.vo.RoundStateInfo;
import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.StateMachine;
import io.grpc.learning.vo.TaskState;
import io.grpc.learning.vo.TaskZoo;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import com.google.protobuf.ByteString;

import org.tensorflow.Graph;

import static io.grpc.learning.computation.FederatedComp.getGraph;
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
        public int minRequestNum = 2;

        /**
         * @param request
         * @param responseObserver
         */
        @Override
        public void call(ComputationRequest request, StreamObserver<ComputationReply> responseObserver) {
            String nodeName = request.getNodeName();
            String clientId = request.getId();
            if (RoundStateInfo.callRequest.get(nodeName) != null &&
                    RoundStateInfo.callRequest.get(nodeName).contains(clientId)) {
                while (RoundStateInfo.roundState.get(nodeName) == StateMachine.wait) {
                    try {
                        RoundStateInfo.waitRequest.get(nodeName).add(clientId);
                        System.out.println(clientId +" " +StateMachine.wait + "...");
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                RoundStateInfo.waitRequest.get(nodeName).remove(clientId);
                RoundStateInfo.collectValueRequest.clear();
                RoundStateInfo.roundState.put(nodeName, StateMachine.start);
            }
            RoundStateInfo.callUpdate(nodeName, clientId);
            Graph graph = new GraphZoo().getGraphZoo().get(nodeName);
            byte[] byteGraph = graph == null ? getGraph(nodeName).toGraphDef() : graph.toGraphDef();
            ComputationReply.Builder reply = ComputationReply.newBuilder();
            reply.setGraph(ByteString.copyFrom(byteGraph));
            reply.setRound(RoundStateInfo.epochMap.get(nodeName));
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

        /**
         * @param request
         * @param responseObserver
         */
        @Override
        public void callValue(ComputationRequest request, StreamObserver<TensorValue> responseObserver) {
            String nodeName = request.getNodeName();
            String clientId = request.getId();
            HashMap<String, SequenceData> weightMap =
                    ModelWeights.weightsCollector.get(clientId);
            SequenceData weight;
            if (weightMap == null) {
                FederatedComp.weightsInitializer(nodeName, clientId);
            }
            weight = ModelWeights.weightsAggregation.get(nodeName);
            TensorValue.Builder reply = offsetStreamReply(weight, request.getOffset());
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

        /**
         * @param request
         * @param responseObserver
         */
        @Override
        public void sendValue(TensorValue request, StreamObserver<ValueReply> responseObserver) {
            ValueReply.Builder reply;
            StateMachine currentState = RoundStateInfo.roundState.get(request.getNodeName());
            if (StateMachine.end != currentState) {
                RoundStateInfo.collectValueUpdate(request);
                if (RoundStateInfo.collectValueRequest
                        .get(request.getNodeName()).size() == minRequestNum) {
                    // update weights
                    FederatedComp.aggregationInner(request);
                    RoundStateInfo.roundState.put(request.getNodeName(), StateMachine.end);
                    FederatedComp.update = false;
                }
            }
            reply = ValueReply.newBuilder().setMessage(true);
            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }

    }
}
