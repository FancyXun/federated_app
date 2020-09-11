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
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.Enumeration;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.tensorflow.*;

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
        String localIP = "127.0.0.1";
        Enumeration<NetworkInterface> n = NetworkInterface.getNetworkInterfaces();
        try {
            NetworkInterface e = n.nextElement();
            Enumeration<InetAddress> a = e.getInetAddresses();
            a.nextElement();
            InetAddress addr = a.nextElement();
            localIP = addr.getHostAddress();
        }
        catch (Exception e1){
            localIP = "127.0.0.1";
        }

        int port = 50051;
        server = ServerBuilder.forPort(port)
                .addService(new ComputationImpl())
                .build()
                .start();
        logger.info("Server started, ip is "+localIP + " listening on " + port);
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
    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException, IllegalAccessException, InstantiationException {
        final ComputationServer server = new ComputationServer();
        server.start();
        server.blockUntilShutdown();
    }

    static class ComputationImpl extends ComputationGrpc.ComputationImplBase {

        @Override
        public void call(ComputationRequest req, StreamObserver<ComputationReply> responseObserver) {
            String clientId = req.getId();
            String node_name = req.getNodeName();
            Graph graph = new Graph();
            try {
                logger.info("Server received request " + url + "." + node_name + " from " + clientId);
                ClassLoader classLoader = Class.forName(url + "." + node_name).getClassLoader();
                BaseGraph basegraph = (BaseGraph) classLoader.loadClass(url + "." + node_name).newInstance();
                graph = basegraph.getGraph();
            } catch (Exception ClassNotFoundException) {

            }
            byte[] byteGraph = graph.toGraphDef();
            ComputationReply.Builder reply = ComputationReply.newBuilder();
            reply.setMessage("Received request from " + clientId);
            reply.setGraph(ByteString.copyFrom(byteGraph));

            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }
    }
}
