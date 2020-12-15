package io.grpc.learning.model;

import org.tensorflow.Graph;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;


public class Initializer {

    private LinkedHashMap<String, String> modelMap;
    private LinkedHashMap<String, String> modelInitMap;
    private LinkedHashMap<String, String> metaMap;
    private Graph graph;

    public LinkedHashMap<String, String> getModelMap() {
        return modelMap;
    }

    public LinkedHashMap<String, String> getModelInitMap() {
        return modelInitMap;
    }

    public LinkedHashMap<String, String> getMetaMap() {
        return metaMap;
    }

    public Graph getGraph() {
        return graph;
    }

    private static class InitializerHolder {
        private static Initializer instance = new Initializer();
    }

    public Initializer() {
    }

    public static Initializer getInstance() {
        return InitializerHolder.instance;
    }

    public void loadModel() {

        graph = new Graph();
        InputStream modelStream = null;
        String var2 = "resource/modelMeta/sphere2.pb";
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
        String trainable_var = "resource/modelMeta/sphere2_trainable_var.txt";
        String trainable_init_var = "resource/modelMeta/sphere2_trainable_init_var.txt";
        String feed_fetch_var = "resource/modelMeta/sphere2_feed_fetch.txt";
        modelMap = loadModelMeta(trainable_var);
        modelInitMap = loadModelMeta(trainable_init_var);
        metaMap = loadModelMeta(feed_fetch_var);
    }


    private LinkedHashMap<String, String> loadModelMeta(String filePath) {

        LinkedHashMap<String, String> map = new LinkedHashMap<String, String>();

        String line;
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(filePath));
            int emptyLine = 0;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(";", 2);
                if (parts.length >= 2) {
                    String key = parts[0];
                    String value = parts[1];
                    map.put(key, value);
                } else {
                    map.put(String.valueOf(emptyLine), "null");
                    emptyLine += 1;
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }
}
