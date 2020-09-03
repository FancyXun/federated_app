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

import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.UUID;
import org.tensorflow.*;

/**
 * A simple client that requests a computation from the {@link ComputationServer}.
 */
public class ComputationClient {
  private static final Logger logger = Logger.getLogger(ComputationClient.class.getName());

  private final ComputationGrpc.ComputationBlockingStub blockingStub;

  public ComputationClient(Channel channel) {
    // 'channel' here is a Channel, not a ManagedChannel, so it is not this code's responsibility to
    // shut it down.

    // Passing Channels to code makes code easier to test and makes it easier to reuse Channels.
    blockingStub = ComputationGrpc.newBlockingStub(channel);
  }

  public void call(String id) {
    logger.info("Will try to call " + id + " ...");
    ComputationRequest request = ComputationRequest.newBuilder().setId(id).build();
    ComputationReply response;
    try {
      response = blockingStub.call(request);
    } catch (StatusRuntimeException e) {
      logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
      return;
    }
    logger.info("calling: " + response.getMessage());
    Graph graph = new Graph();
    graph.importGraphDef(response.getGraph().toByteArray());
    Session session = new Session(graph);
    Tensor tensor = session.runner().fetch("xy").feed("x", Tensor.create(5.0f)).feed("y", Tensor.create(2.0f)).run().get(0);
    System.out.println(tensor.floatValue());
  }

  public static void main(String[] args) throws Exception {
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
    try {
      ComputationClient client = new ComputationClient(channel);
      client.call(local_id);
    } finally {
      // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
      // resources the channel should be shut down when it will no longer be used. If it may be used
      // again leave it running.
      channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
  }
}
