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
    private String pyDir = (String) rb.getObject("pyRootPath");

    // model info
    private String graphPath = (String) rb.getObject("graphPath");
    private String graphGlobalVarPath = (String) rb.getObject("graphGlobalVarPath");
    private String graphTrainableVarPath = (String) rb.getObject("graphTrainableVarPath");
    private String graphTrainInfoPath = (String) rb.getObject("graphTrainInfoPath");
    private String oneHot = (String) rb.getObject("oneHot");
    private String dataUrl = (String) rb.getObject("dataUrl");
    private int labelNum = Integer.parseInt(String.valueOf(rb.getObject("labelNum")));
    private int height = Integer.parseInt(String.valueOf(rb.getObject("height")));
    private int width = Integer.parseInt(String.valueOf(rb.getObject("width")));

    public String getOneHot() {
        return oneHot;
    }

    public int getLabelNum() {
        return labelNum;
    }

    public String getDataUrl() {
        return dataUrl;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public HashMap<String ,Float>  hashMapACC = new HashMap<>();
    public HashMap<String ,Float>  hashMapLoss = new HashMap<>();
    public HashMap<String ,Integer>  hashMapDataNum = new HashMap<>();


    public String getModelWeighsPath() {
        return modelWeighsPath;
    }

    public void setModelWeighsPath(String modelWeighsPath) {
        this.modelWeighsPath = modelWeighsPath;
    }

    private String modelWeighsPath = (String) rb.getObject("modelWeighsPath");

    // agg
    private final String pyDirAgg = (String) rb.getObject("pyAggRootPath");
    private Graph graph;

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

    public void gen_graph() {
        Process process;
        try {
            System.out.println(String.format("%s %s ", pythonExe, pyDir));
            process = Runtime.getRuntime().exec(String.format("%s %s ", pythonExe, pyDir));
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
            process = Runtime.getRuntime().exec(String.format("%s %s %s",
                    pythonExe, pyDirAgg, modelWeighsPath));
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
        gen_graph();
        graph = new Graph();
        InputStream modelStream = null;

        try {
            modelStream = new FileInputStream(graphPath);
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

        modelTrainableMap = loadModelMeta(graphTrainableVarPath);
        modelInitMap = loadModelMeta(graphGlobalVarPath);
        metaMap = loadModelMeta(graphTrainInfoPath);
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

    public void LoadModelWeights() {
        layerWeightsHashMap.clear();
        layerWeightsShapeHashMap.clear();
        layerWeightsInitHashMap.clear();
        LinkedHashMap<String, String> modelTrainableMap = this.getModelTrainableMap();
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
            try (BufferedReader br = new BufferedReader(new FileReader(modelWeighsPath+"/"+"average/"+
                    modelTrainableMap.get(key).replace("/","_")+".txt"))) {

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

    public void calMetrics(){
        float acc_agg = 0;
        float loss_agg = 0;
        int dataSum = 0;
        for (String key: hashMapACC.keySet()){
            acc_agg += hashMapACC.get(key) * hashMapDataNum.get(key);
        }
        for (String key: hashMapLoss.keySet()){
            loss_agg += hashMapLoss.get(key) * hashMapDataNum.get(key);
        }
        for (int num: hashMapDataNum.values()){
            dataSum +=num;
        }
        System.out.println("loss: " + loss_agg/dataSum);
        System.out.println("acc: " + acc_agg/dataSum);
    }
}
