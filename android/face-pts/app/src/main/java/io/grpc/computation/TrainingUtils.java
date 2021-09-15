package io.grpc.computation;

import android.content.Context;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.LinkedHashMap;

@Deprecated
public class TrainingUtils {

    public void localTraining(Context context, String pbPath) {
        /**
         * just for local training debug in android
         */
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            boolean var1 = pbPath.startsWith("file:///android_asset/");
            String var2 = var1 ? pbPath.split("file:///android_asset/")[1] : pbPath;
            modelStream = context.getAssets().open(var2);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] byteArray = buffer.toByteArray();
            graph.importGraphDef(byteArray);
            Iterator<Operation> operationIterator = graph.operations();
            Session session = new Session(graph);
            String trainable_var = "file:///android_asset/protobuffer/inception_resnet_trainable_var.txt";
            String feed_fetch_var = "file:///android_asset/protobuffer/inception_resnet_feed_fetch.txt";
            String data_path = "sampleData/casiaWebFace";
            LinkedHashMap<String, String> modelMap = loadModelMeta(context, trainable_var);
            LinkedHashMap<String, String> metaMap = loadModelMeta(context, feed_fetch_var);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public LinkedHashMap<String, String> loadModelMeta(Context context, String filePath) {
        LinkedHashMap<String, String> map = new LinkedHashMap<String, String>();
        boolean var1 = filePath.startsWith("file:///android_asset/");
        String var2 = var1 ? filePath.split("file:///android_asset/")[1] : filePath;
        String line;
        BufferedReader reader = null;
        try {
            InputStream modelStream = context.getAssets().open(var2);
            reader = new BufferedReader(new InputStreamReader(modelStream));
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
