package io.grpc.learning.model;

import org.jetbrains.bio.npy.NpzFile;
import org.tensorflow.Graph;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.ResourceBundle;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import computation.TensorEntity;
import io.grpc.learning.computation.LayerWeights;
import io.grpc.learning.computation.ModelWeights;


public class ModelHelper {

    private LinkedHashMap<String, String> modelTrainableMap;
    private LinkedHashMap<String, String> modelInitMap;
    private LinkedHashMap<String, String> metaMap;
    private ResourceBundle rb = ResourceBundle.getBundle("resource", Locale.getDefault());
    private String pythonExe = (String) rb.getObject("pythonExe");
    private String rootPath = (String) rb.getObject("pyRootPath");
    private final String var2 = rootPath + "sphere_unfrozen.pb";
    private final String pyDir = rootPath + "sphere_frozen_arg.py";
    private final String trainable_var = rootPath + "sphere2_trainable_var_unfrozen.txt";
    private final String  trainable_init_var = rootPath + "sphere2_trainable_init_var_unfrozen.txt";
    private final String  feed_fetch_var = rootPath + "sphere2_feed_fetch_unfrozen.txt";
    private final String pyDirAgg = (String) rb.getObject("pyAggRootPath");
    private Graph graph;
    private int blockInit;

    private HashMap <String, LayerWeights.Builder> layerWeightsHashMap = new HashMap<>();
    private HashMap <String, String> layerWeightsShapeHashMap = new HashMap<>();

    public HashMap<String, String> getLayerWeightsInitHashMap() {
        return layerWeightsInitHashMap;
    }

    private HashMap <String, String> layerWeightsInitHashMap = new HashMap<>();

    public HashMap<String, String> getLayerWeightsShapeHashMap() {
        return layerWeightsShapeHashMap;
    }

    public HashMap<String, LayerWeights.Builder> getLayerWeightsHashMap() {
        return layerWeightsHashMap;
    }



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
        private static ModelHelper instance = new ModelHelper();
    }

    public ModelHelper() {
    }

    public static ModelHelper getInstance() {
        return InitializerHolder.instance;
    }

    public void gen_graph(int block) {
        Process process;
        try {
            System.out.println(String.format("%s %s -p %s --unfrozen=%s", pythonExe, pyDir, rootPath, 0));
            process = Runtime.getRuntime().exec(String.format("%s %s -p %s --unfrozen=%s", pythonExe, pyDir, rootPath, 0));
            BufferedReader in = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            while ((line = in.readLine()) != null) {
                if (line.contains("error")){
                    System.out.println(line);
                }
            }
            in.close();
            process.waitFor();
            process.destroy();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void updateWeights(){
        Process process;
        try {
            process = Runtime.getRuntime().exec(String.format("%s %s", pythonExe, pyDirAgg));
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

    public void loadModel(int block) {
        gen_graph(block);
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

    public void ModelWeightsUpdate() {
        layerWeightsHashMap.clear();
        layerWeightsShapeHashMap.clear();
        layerWeightsInitHashMap.clear();
        ModelHelper modelHelper = ModelHelper.getInstance();
        LinkedHashMap<String, String> modelTrainableMap = modelHelper.getModelTrainableMap();
        Pattern p = Pattern.compile("\\d+");
        ModelWeights.Builder modelWeightsBuilder = ModelWeights.newBuilder();
        ArrayList<TensorEntity.TensorShapeProto.Builder> tensorShapeArrayList = new ArrayList<>();
        int layer_index = 0;
        for (String key : modelTrainableMap.keySet()) {
            // todo: np array read error
//            Path filePath = new File("/tmp/model_weights/average/layer_"+layer_index+".npz").toPath();
//            NpzFile.Reader reader = NpzFile.read(filePath);
//            System.out.println("layer_index" + layer_index);
//            float[] floats = reader.get("arr_0", reader.introspect().get(0).getShape()[0]).asFloatArray();
            TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                    TensorEntity.TensorShapeProto.newBuilder();
            String shape = modelInitMap.get(key);
            Matcher m = p.matcher(shape);
            int size = 1;
            int dim_index = 0;
            while (m.find()) {
                int dim = Integer.parseInt(m.group());
                size = size * dim;
                TensorEntity.TensorShapeProto.Dim.Builder dimBuilder =
                        TensorEntity.TensorShapeProto.Dim.newBuilder();
                dimBuilder.setSize(dim);
                tensorShapeBuilder.addDim(dim_index, dimBuilder);
                dim_index++;
            }

            TensorEntity.TensorProto.Builder tensorBuilder =
                    TensorEntity.TensorProto.newBuilder();
            LayerWeights.Builder layerWeightsBuilder = LayerWeights.newBuilder();
            try (BufferedReader br = new BufferedReader(new FileReader("/tmp/model_weights/average/"+
                    modelTrainableMap.get(key).replace("/","@")+".txt"))) {
                try {
                    String line = br.readLine();
                    while (line != null) {
                        tensorBuilder.addFloatVal(Float.parseFloat(line));
                        line = br.readLine();
                    }
                } finally {
                    br.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            /*
             * To create DataInputStream object, use
             * DataInputStream(InputStream in) constructor.
             */

            tensorBuilder.setTensorShape(tensorShapeBuilder);
            tensorShapeArrayList.add(tensorShapeBuilder);
            modelWeightsBuilder.addTensor(layer_index, tensorBuilder);
            layerWeightsBuilder.setTensor(tensorBuilder);
            layerWeightsBuilder.setLayerId(layer_index);
            layerWeightsHashMap.put(modelTrainableMap.get(key), layerWeightsBuilder);
            layerWeightsShapeHashMap.put(modelTrainableMap.get(key),shape);
            layerWeightsInitHashMap.put(modelTrainableMap.get(key), key);
            layer_index++;
        }
    }
}
