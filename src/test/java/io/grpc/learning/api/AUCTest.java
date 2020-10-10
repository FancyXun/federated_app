package io.grpc.learning.api;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Iterator;

import sun.misc.IOUtils;

public class AUCTest {

    @Test
    public void AUCGraphTest(){
        Graph graph = new Graph();
        InputStream modelStream;
        float[] var = new float[20];
        float[] var1 = new float[20];
        float[] var2 = new float[20];
        float[] var3 = new float[20];
        float[] pre = {0,0,0,0,1,1,1,1};
        float[] subset = Arrays.copyOfRange(pre, 3, 5);
        float[] labels =  {0.3f, 0.4f, 0.6f, 0.2f, 0.7f,0.5f, 0.1f, 0.9f};
        try {
            modelStream = new FileInputStream("/home/zhangxun/code/android_demo/federated_app" +
                    "/android/framework/app/src/main/assets/metrics/auc.pb");
            graph.importGraphDef(IOUtils.readAllBytes(modelStream));
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
        Iterator<Operation> operationIterator = graph.operations();
        while (operationIterator.hasNext()){
            Operation op = operationIterator.next();
            System.out.println(op);
        }
        session.runner()
                .feed("auc_pair/true_positives/Initializer/zeros", Tensor.create(var))
                .addTarget("auc_pair/true_positives/Assign")
                .run();

        session.runner()
                .feed("auc_pair/false_positives/Initializer/zeros", Tensor.create(var))
                .addTarget("auc_pair/false_positives/Assign")
                .run();

        session.runner()
                .feed("auc_pair/true_negatives/Initializer/zeros", Tensor.create(var))
                .addTarget("auc_pair/true_negatives/Assign")
                .run();

        session.runner()
                .feed("auc_pair/false_negatives/Initializer/zeros", Tensor.create(var))
                .addTarget("auc_pair/false_negatives/Assign")
                .run();
        Tensor tensor = session.runner()
                .feed("prediction", Tensor.create(pre))
                .feed("labels", Tensor.create(labels)).fetch("auc_pair/update_op").run().get(0);
        System.out.println(tensor.floatValue());
    }

    @Test
    public void AUCImport(){
        Graph graph = new Graph();
        Graph graph1 = new Graph();
        InputStream modelStream = null;


        try {
            String var2 = "/home/zhangxun/code/android_demo/federated_app/" +
                    "android/framework/app/src/main/assets/metrics/auc.pb";
            modelStream = new FileInputStream(var2);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] byteArray = buffer.toByteArray();
            graph.importGraphDef(byteArray);
            graph1.importGraphDef(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
    }

}
