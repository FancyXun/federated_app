package io.grpc.learning.model;

import org.tensorflow.Graph;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.ResourceBundle;


public class Initializer {

    private LinkedHashMap<String, String> modelTrainableMap;
    private LinkedHashMap<String, String> modelInitMap;
    private LinkedHashMap<String, String> metaMap;
    private ResourceBundle rb = ResourceBundle.getBundle("resource", Locale.getDefault());
    private String pythonExe = (String) rb.getObject("pythonExe");
    private String rootPath = (String) rb.getObject("pyRootPath");
    private final String var2 = rootPath + "sphere_frozen123.pb";
    private final String pyDir = rootPath + "sphere_frozen_arg.py";
    private final String trainable_var = rootPath + "sphere2_trainable_var_f123.txt";
    private final String  trainable_init_var = rootPath + "sphere2_trainable_init_var_f123.txt";
    private final String  feed_fetch_var = rootPath + "sphere2_feed_fetch_f123.txt";
    private Graph graph;


    public LinkedHashMap<String, String> getModelTrainableMap() {
        return modelTrainableMap;
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

    public void gen_graph(int block) {
        Process process;
        try {
            process = Runtime.getRuntime().exec(String.format("%s %s -p %s --unfrozen=%s", pythonExe, pyDir, rootPath, block));
            BufferedReader in = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            process.destroy();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void loadModel() {
        gen_graph(1);
        graph = new Graph();
        InputStream modelStream = null;

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

        modelTrainableMap = loadModelMeta(trainable_var);
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
