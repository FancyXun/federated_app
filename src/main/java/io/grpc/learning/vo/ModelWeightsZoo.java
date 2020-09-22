package io.grpc.learning.vo;

import java.util.HashMap;

public class ModelWeightsZoo {

    private volatile static ModelWeightsZoo instance = null;

    public static ModelWeightsZoo getInstance() {
        if (instance == null) {
            synchronized (ModelWeightsZoo.class) {
                if (instance == null) {
                    instance = new ModelWeightsZoo();
                }
            }

        }
        return instance;
    }

    public HashMap<String, SequenceData> getModelZoo() {
        return modelZoo;
    }

    public void setModelZoo(HashMap<String, SequenceData> modelZoo) {
        ModelWeightsZoo.modelZoo = modelZoo;
    }

    private static HashMap<String, SequenceData> modelZoo = new HashMap<>();
}
