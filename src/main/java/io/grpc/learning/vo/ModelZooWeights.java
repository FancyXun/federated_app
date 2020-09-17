package io.grpc.learning.vo;

import java.util.HashMap;

public class ModelZooWeights {

    private volatile static ModelZooWeights instance = null;

    public static ModelZooWeights getInstance() {
        if (instance == null) {
            synchronized (ModelZooWeights.class) {
                if (instance == null) {
                    instance = new ModelZooWeights();
                }
            }

        }
        return instance;
    }

    public HashMap<String, SequenceData> getModelZoo() {
        return modelZoo;
    }

    public void setModelZoo(HashMap<String, SequenceData> modelZoo) {
        ModelZooWeights.modelZoo = modelZoo;
    }

    private static HashMap<String, SequenceData> modelZoo = new HashMap<>();
}
