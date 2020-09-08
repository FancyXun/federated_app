package io.grpc.learning.api;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import sun.misc.IOUtils;

public class LogisticsRegressionTests {

    @Test
    public void LogisticsRegressionTest(){
        LogisticsRegression logisticsRegression = new LogisticsRegression(true,"src/main/python/LogisticsRegression.py");
        float [][] x = {{1f,1f}, {2f, 2f}};
        float [] y = {1f, 2f};
        float [] b = new float[2];
        float [][] w = new float[2][2];
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            modelStream = new FileInputStream(logisticsRegression.pbPath);
            graph.importGraphDef(IOUtils.readAllBytes(modelStream));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
        Iterator<Operation> operationIterator = graph.operations();
        session.runner()
                .feed("w/init", Tensor.create(w))
                .addTarget("w/Assign")
                .run();
        session.runner()
                .feed("b/init", Tensor.create(b))
                .addTarget("b/Assign")
                .run();
        Tensor tensor = session.runner().fetch("cost").feed("x", Tensor.create(x)).feed("y", Tensor.create(y)).run().get(0);
        System.out.println(String.valueOf(tensor.floatValue()));
        logisticsRegression.deletePBFile();
    }

}
