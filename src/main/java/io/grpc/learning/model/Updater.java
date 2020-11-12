package io.grpc.learning.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import computation.TensorEntity;
import io.grpc.learning.computation.LayerWeights;
import io.grpc.learning.computation.ModelWeights;
import io.grpc.learning.logging.SystemOut;

public class Updater {

    public ModelWeights.Builder modelWeightsBuilder;
    public ArrayList<LayerWeights.Builder> layerWeightsArrayList;
    public ArrayList<TensorEntity.TensorShapeProto.Builder> tensorShapeArrayList;
    public LinkedHashMap<String, LinkedHashMap<Long, List<Float>>> weightsLinkedHashMap;

    private void ModelWeightsInitializer() {
        Initializer initializer = Initializer.getInstance();
        LinkedHashMap<String, String> modelMap = initializer.getModelMap();
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

    public void updateWeights(){
        // todo: add more clients to do
        modelWeightsBuilder.clear();
        modelWeightsBuilder = ModelWeights.newBuilder();
        for(String key: weightsLinkedHashMap.keySet()){
            LinkedHashMap<Long, List<Float>> listLinkedHashMap = weightsLinkedHashMap.get(key);
            int layer_index =0;
            for (Long layer_id: listLinkedHashMap.keySet()){
                TensorEntity.TensorProto.Builder tensorBuilder =
                        TensorEntity.TensorProto.newBuilder();
                tensorBuilder.setTensorShape(tensorShapeArrayList.get(layer_index));
                for(int i =0; i< listLinkedHashMap.get(layer_id).size(); i++){
                    tensorBuilder.addFloatVal(listLinkedHashMap.get(layer_id).get(i));
                }
                modelWeightsBuilder.addTensor(layer_index, tensorBuilder);
                layer_index ++;
            }
        }
        weightsLinkedHashMap.clear();
    }

    public Updater() {
        this.ModelWeightsInitializer();
    }

    private static class UpdaterHolder {
        private static Updater instance = new Updater();
    }

    public static Updater getInstance() {
        return Updater.UpdaterHolder.instance;
    }
}
