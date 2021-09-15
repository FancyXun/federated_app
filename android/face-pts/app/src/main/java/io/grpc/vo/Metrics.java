package io.grpc.vo;


import java.util.ArrayList;
import java.util.List;

public class Metrics {
    public String model;
    private int round;
    public List<String> metricsName;
    public List<Float> metrics;
    public int weights;

    public Metrics(){
        this.metricsName = new ArrayList<>();
        this.metrics = new ArrayList<>();
    }
}
