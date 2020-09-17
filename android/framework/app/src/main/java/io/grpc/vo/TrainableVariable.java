package io.grpc.vo;

import java.util.HashMap;
import java.util.List;

public class TrainableVariable {
    private HashMap<String, float[][]> weight;
    private HashMap<String, float[]> bias;
    private HashMap<String, List> weightShape;
    private HashMap<String, Boolean> trainable;
    private HashMap<String, String> nameMap;

    public List<String> getBackPropagationList() {
        return backPropagationList;
    }

    public void setBackPropagationList(List<String> backPropagationList) {
        this.backPropagationList = backPropagationList;
    }

    private List<String>  backPropagationList;

    public HashMap<String, String> getNameMap() {
        return nameMap;
    }

    public void setNameMap(HashMap<String, String> nameMap) {
        this.nameMap = nameMap;
    }



    public HashMap<String, float[][]> getWeight() {
        return weight;
    }

    public void setWeight(HashMap<String, float[][]> weight) {
        this.weight = weight;
    }

    public HashMap<String, float[]> getBias() {
        return bias;
    }

    public void setBias(HashMap<String, float[]> bias) {
        this.bias = bias;
    }


}
