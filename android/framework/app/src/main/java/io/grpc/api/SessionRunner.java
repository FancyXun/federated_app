package io.grpc.api;

import org.tensorflow.Graph;

public class SessionRunner {
    private Graph graph;

    public SessionRunner(Graph graph) {
        this.graph = graph;
    }

    public void invoke() {

    }
}
