package io.grpc.learning.storage;

import java.util.HashMap;

import io.grpc.learning.computation.Metrics;

public class MetricsContainer {

    private volatile static MetricsContainer instance = null;

    public static MetricsContainer getInstance() {
        if (instance == null) {
            synchronized (MetricsContainer.class) {
                if (instance == null) {
                    instance = new MetricsContainer();
                }
            }

        }
        return instance;
    }

    public static HashMap<String, HashMap<String, Metrics>> MetricsCollector = new HashMap<>();
}
