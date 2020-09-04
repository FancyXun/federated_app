package io.grpc.learning.api;

import org.tensorflow.Graph;

public abstract class BaseGraph {
    protected Graph graph;

    public Graph getGraph() {
        return this.graph;
    }
}
