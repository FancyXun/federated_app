package io.grpc.learning.vo;

import java.util.HashMap;

public class ModelWeights {

    private volatile static ModelWeights instance = null;

    public static ModelWeights getInstance() {
        if (instance == null) {
            synchronized (ModelWeights.class) {
                if (instance == null) {
                    instance = new ModelWeights();
                }
            }

        }
        return instance;
    }


    public static HashMap<String, SequenceData> weightsAggregation = new HashMap<>();
    public static HashMap<String, HashMap<String, SequenceData>> weightsCollector = new HashMap<>();
}
