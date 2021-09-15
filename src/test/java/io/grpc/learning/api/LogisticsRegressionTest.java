package io.grpc.learning.api;

import com.alibaba.fastjson.JSON;

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
import java.util.List;

import io.grpc.learning.utils.JsonUtils;
import io.grpc.learning.utils.LocalCSVReader;
import io.grpc.learning.vo.TensorVarName;
import sun.misc.IOUtils;

public class LogisticsRegressionTest {

    @Test
    public void LogisticsRegressionTest() {
        LogisticsRegression logisticsRegression = new LogisticsRegression(false);
        float[][] x = {{1f, 1f}, {2f, 2f}};
        float[] y = {1f, 2f};
        float[] b = new float[2];
        float[][] w = new float[141][2];
        float batchSize = 10f;
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            modelStream = new FileInputStream(logisticsRegression.pbPath);
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
                .feed("w/zero", Tensor.create(w))
                .addTarget("w/Variable/Assign")
                .run();
        session.runner()
                .feed("b/zero", Tensor.create(b))
                .addTarget("b/Variable/Assign")
                .run();
        Tensor tensor = session.runner().fetch("loss").feed("x", Tensor.create(x)).feed("y", Tensor.create(y))
                .feed("batch_size", Tensor.create(batchSize)).run().get(0);
        System.out.println(tensor.floatValue());
        logisticsRegression.deletePBFile();
    }

    @Test
    public void LogisticsRegressionTrainingTest() {
        String dataPath = "src/main/resources/test_data/bank_zhongyuan/test_data1.csv";
        LocalCSVReader localCSVReader = new LocalCSVReader(dataPath, 0, "target");
        LogisticsRegression logisticsRegression = new LogisticsRegression(false);
        float[][] x = localCSVReader.getX();
        float[][] y = localCSVReader.getY_oneHot();
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
        String s = JsonUtils.readJsonFile(logisticsRegression.pbPath.replace(".pb",".json"));
        TensorVarName tensorVarName = JsonUtils.jsonToMap(JSON.parseObject(s));

        for (int i = 0; i < tensorVarName.getTensorName().size(); i++) {
            List<Integer> integerList = tensorVarName.getTensorShape().get(i);
            Tensor tensor = null;
            if (integerList.size() == 1) {
                tensor = Tensor.create(new float[integerList.get(0)]);
            }
            if (integerList.size() == 2) {
                tensor = Tensor.create(new float[integerList.get(0)][integerList.get(1)]);
            }
            session.runner()
                    .feed(tensorVarName.getTensorName().get(i), tensor)
                    .addTarget(tensorVarName.getTensorTargetName().get(i)).run();
        }
        for (int i = 0; i < 10; i++) {
            Tensor tensor = session.runner()
                    .fetch("loss")
                    .feed(tensorVarName.getPlaceholder().get(1), Tensor.create(x))
                    .feed(tensorVarName.getPlaceholder().get(2), Tensor.create(y)).run().get(0);

            for (int j = 0; j < tensorVarName.getTensorAssignName().size(); j++) {
                Tensor t = session.runner().fetch(tensorVarName.getTensorAssignName().get(j))
                        .feed(tensorVarName.getPlaceholder().get(0), Tensor.create(batchSize))
                        .feed(tensorVarName.getPlaceholder().get(1), Tensor.create(x))
                        .feed(tensorVarName.getPlaceholder().get(2), Tensor.create(y))
                        .run().get(0);
                System.out.println("*");

            }
            System.out.println("loss:" + tensor.floatValue());

        }


        logisticsRegression.deletePBFile();
    }

}
