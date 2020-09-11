package io.grpc.api;

import java.util.HashMap;

import io.grpc.utils.LocalCSVReader;

public class TwoDimEntity implements GraphEntity {
    private float[][] x;
    private float[][] y_oneHot;
    private float[] y;
    private HashMap<String, float[][]> weight;
    private HashMap<String, float[]> bias;

    public TwoDimEntity(LocalCSVReader localCSVReader){
       this.x = localCSVReader.getX();
       this.y_oneHot= localCSVReader.getY_oneHot();
       this.y= localCSVReader.getY();
    }

    @Override
    public void initVar() {

    }

}
