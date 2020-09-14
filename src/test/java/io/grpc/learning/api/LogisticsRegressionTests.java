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

import io.grpc.learning.utils.LocalCSVReader;
import sun.misc.IOUtils;

public class LogisticsRegressionTests {

    @Test
    public void LogisticsRegressionTest(){
        LogisticsRegression logisticsRegression = new LogisticsRegression(false);
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
        System.out.println(tensor.floatValue());
        logisticsRegression.deletePBFile();
    }

    @Test
    public void LogisticsRegressionTrainingTest(){
        String dataPath = "src/main/resources/test_data/bank_zhongyuan/test_data1.csv";
        LocalCSVReader localCSVReader = new LocalCSVReader(dataPath, 0,"target");
        LogisticsRegression logisticsRegression = new LogisticsRegression(false);
        float [][] x = localCSVReader.getX();
        float [][] y = localCSVReader.getY_oneHot();
        float [] b = new float[localCSVReader.getY_oneHot()[0].length];
        float [][] w = new float[localCSVReader.getX()[0].length][localCSVReader.getY_oneHot()[0].length];
        float [][] copy = new float[localCSVReader.getX()[0].length][localCSVReader.getY_oneHot()[0].length];
        float batchSize = x.length;
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
        session.runner().feed("w/init", Tensor.create(w)).addTarget("w/Assign").run();
        session.runner().feed("b/init", Tensor.create(b)).addTarget("b/Assign").run();
        for (int i = 0; i < 5; i++){
            Tensor tensor = session.runner().fetch("cost").feed("x", Tensor.create(x))
                    .feed("y", Tensor.create(y)).run().get(0);
            session.runner().feed("x", Tensor.create(x))
                    .feed("y", Tensor.create(y))
                    .feed("batch_size",Tensor.create(batchSize)).addTarget("w_assign").run();
            session.runner().feed("x", Tensor.create(x))
                    .feed("y", Tensor.create(y))
                    .feed("batch_size",Tensor.create(batchSize)).addTarget("b_assign").run();
            Tensor tensor1 = session.runner().fetch("w_assign")
                    .feed("x", Tensor.create(x))
                    .feed("y", Tensor.create(y))
                    .feed("batch_size",Tensor.create(batchSize)).run().get(0);
            tensor1.copyTo(copy);
            byte [] tmp = Tensor.create(copy).bytesValue();
            Tensor.create(b).bytesValue();
            System.out.println(copy[0][0]);
            System.out.println("cost:" + tensor.floatValue());

        }


        logisticsRegression.deletePBFile();
    }

}
