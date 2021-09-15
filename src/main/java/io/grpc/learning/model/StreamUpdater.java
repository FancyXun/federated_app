package io.grpc.learning.model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.JOptionPane;

import computation.TensorEntity;
import io.grpc.learning.computation.LayerWeights;
import io.grpc.learning.computation.ModelWeights;

public class StreamUpdater {
    public ModelWeights.Builder modelWeightsBuilder;
    public ArrayList<LayerWeights.Builder> layerWeightsArrayList;
    public ArrayList<TensorEntity.TensorShapeProto.Builder> tensorShapeArrayList;
    public LinkedHashMap<String, LinkedHashMap<Long, List<Float>>> weightsLinkedHashMap;
    public LinkedHashMap<Long, List<Float>> modelWeights;
    public String weightsURL = "resource/modelMeta/weights.txt";

    private void ModelWeightsInitializer() {
        // 初始化
        ModelHelper modelHelper = ModelHelper.getInstance();
        LinkedHashMap<String, String> modelMap = modelHelper.getModelTrainableMap();
        Pattern p = Pattern.compile("\\d+");
        modelWeightsBuilder = ModelWeights.newBuilder();
        layerWeightsArrayList = new ArrayList<>();
        tensorShapeArrayList = new ArrayList<>();
        int layer_index = 0;
        for (String key : modelMap.keySet()) {
            TensorEntity.TensorProto.Builder tensorBuilder =
                    TensorEntity.TensorProto.newBuilder();
            TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                    TensorEntity.TensorShapeProto.newBuilder();
            LayerWeights.Builder layerWeightsBuilder = LayerWeights.newBuilder();
            String shape = modelMap.get(key);
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
            for (int i = 0; i < size; i++) {
                tensorBuilder.addFloatVal(0);
            }
            tensorBuilder.setTensorShape(tensorShapeBuilder);
            tensorShapeArrayList.add(tensorShapeBuilder);
            modelWeightsBuilder.addTensor(layer_index, tensorBuilder);
            layerWeightsBuilder.setTensor(tensorBuilder);
            layerWeightsBuilder.setLayerId(layer_index);
            layerWeightsArrayList.add(layerWeightsBuilder);
            layer_index++;
        }
        weightsLinkedHashMap = new LinkedHashMap<>();
    }

    public void updateWeights() {
        // todo: add more clients to do
        modelWeightsBuilder.clear();
        modelWeightsBuilder = ModelWeights.newBuilder();
        this.aggregateWeights();
        int layer_index = 0;
        for (Long layer_id : modelWeights.keySet()) {
            TensorEntity.TensorProto.Builder tensorBuilder =
                    TensorEntity.TensorProto.newBuilder();
            tensorBuilder.setTensorShape(tensorShapeArrayList.get(layer_index));
            for (int i = 0; i < modelWeights.get(layer_id).size(); i++) {
                tensorBuilder.addFloatVal(modelWeights.get(layer_id).get(i));
            }
            modelWeightsBuilder.addTensor(layer_index, tensorBuilder);
            layer_index++;

        }
        weightsLinkedHashMap.clear();
    }

    private void aggregateWeights() {
        int idx = 0;
        for (String key : weightsLinkedHashMap.keySet()) {
            if (idx == 0) {
                modelWeights = weightsLinkedHashMap.get(key);
            } else {
                LinkedHashMap<Long, List<Float>> listLinkedHashMap = weightsLinkedHashMap.get(key);
                for (Long layer_id : listLinkedHashMap.keySet()) {
                    for (int i = 0; i < listLinkedHashMap.get(layer_id).size(); i++) {
                        float ele = modelWeights.get(layer_id).get(i) +
                                listLinkedHashMap.get(layer_id).get(i);
                        modelWeights.get(layer_id).set(i, ele);
                    }
                }
            }
            idx++;
        }
        for (Long layer_id : modelWeights.keySet()) {
            for (int i = 0; i < modelWeights.get(layer_id).size(); i++) {
                float ele = modelWeights.get(layer_id).get(i) / idx;
                modelWeights.get(layer_id).set(i, ele);
            }
        }
        saveModel();
    }

    private void saveModel() {
        File file = new File(weightsURL);
        try {
            FileWriter fw = new FileWriter(file);
            BufferedWriter output = new BufferedWriter(fw);
            for (Long layer_id : modelWeights.keySet()) {
                output.write(modelWeights.get(layer_id).toString());
                output.newLine();
            }
            output.close();

        } catch (Exception e) {
            JOptionPane.showMessageDialog(null, "I cannot create that file");
        }
    }

    public StreamUpdater() {
        this.ModelWeightsInitializer();
    }

    private static class UpdaterHolder {
        private static StreamUpdater instance = new StreamUpdater();
    }

    public static StreamUpdater getInstance() {
        return StreamUpdater.UpdaterHolder.instance;
    }
}
