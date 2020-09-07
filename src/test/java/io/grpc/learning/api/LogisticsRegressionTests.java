package io.grpc.learning.api;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import sun.misc.IOUtils;

public class LogisticsRegressionTests {

    @Test
    public void loadGraph() throws IOException {
        Float [][] x = new Float[2][10];
        Float [] y = new Float[10];
        Float [] b = new Float[10];
        Float [][] w = new Float[10][10];
        Graph graph = new Graph();
        System.out.println(graph);
        InputStream modelStream = new FileInputStream("src/main/graph/graph.pb");
        graph.importGraphDef(IOUtils.readAllBytes(modelStream));
        Session session = new Session(graph);
        Iterator<Operation> operationIterator = graph.operations();
        while (operationIterator.hasNext()){
            Operation op = operationIterator.next();
            System.out.println(op);
//            if (op.name().equals("w")){
//                session.runner().feed("w", Tensor.create(w));
//            }
//            if (op.name().equals("b")){
//                session.runner().feed("b", Tensor.create(b));
//            }
        }
        Tensor tensor = session.runner().fetch("cost").feed("x", Tensor.create(x)).feed("y", Tensor.create(y)).run().get(0);
        System.out.println(String.valueOf(tensor.floatValue()));
    }

}
