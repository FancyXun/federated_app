package io.grpc.learning.api;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;

public class InceptionResnetTest {

    @Test
    public void readPB() {

        Graph graph = new Graph();
        InputStream modelStream = null;
        String var2 = "resource/modelMeta/inception_resnet.pb";
        try {
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
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
        Iterator<Operation> operationIterator = graph.operations();
        while (operationIterator.hasNext()) {
            Operation op = operationIterator.next();
//            System.out.println(op);
        }
        String trainable_var = "resource/modelMeta/inception_resnet_trainable_var.txt";
        String feed_fetch_var = "resource/modelMeta/inception_resnet_feed_fetch.txt";
        HashMap<String, String> map = readTXT(trainable_var);
        LinkedHashMap<String, String> map1 = readTXT(feed_fetch_var);
    }


    public LinkedHashMap<String, String> readTXT(String filePath) {

        LinkedHashMap<String, String> map = new LinkedHashMap<String, String>();

        String line;
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(filePath));
            int emptyLine = 0;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(":", 2);
                if (parts.length >= 2) {
                    String key = parts[0];
                    String value = parts[1];
                    map.put(key, value);
                } else {
                    System.out.println("ignoring line: " + line);
                    map.put(String.valueOf(emptyLine), "null");
                    emptyLine += 1;
                }
            }

            for (String key : map.keySet()) {
                System.out.println(key + ":" + map.get(key));
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }

}
