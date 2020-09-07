package io.grpc.learning.api;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Tensor;

public class LogisticsRegression extends BaseGraph {

    private boolean loadGraph;


    public LogisticsRegression(boolean loadGraph) {
        this.loadGraph = loadGraph;
        if (this.loadGraph){
            Graph graph = new Graph();
            this.graph = graph;
        }
    }

}
