package io.grpc.learning.api;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Tensor;

public class FloatMul extends BaseGraph{
    public FloatMul(){
        Graph graph = new Graph();
        Operation x = graph.opBuilder("Const", "x")
                .setAttr("dtype", DataType.FLOAT)
                .setAttr("value", Tensor.create(3.0f))
                .build();
        Operation y = graph.opBuilder("Placeholder", "y")
                .setAttr("dtype", DataType.FLOAT)
                .build();
        graph.opBuilder("Mul", "xy")
                .addInput(x.output(0))
                .addInput(y.output(0))
                .build();
        this.graph = graph;
    }
}
