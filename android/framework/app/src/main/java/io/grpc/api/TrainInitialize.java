package io.grpc.api;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import io.grpc.utils.LocalCSVReader;
import io.grpc.utils.listConvert;
import io.grpc.vo.TrainableVariable;

public class TrainInitialize implements GraphInitialize {
    private float[][] x_train;
    private float[][] x_val;
    private float[][] y_oneHot_train;

    public float[][] getX_train() {
        return x_train;
    }

    public void setX_train(float[][] x_train) {
        this.x_train = x_train;
    }

    public float[][] getX_val() {
        return x_val;
    }

    public void setX_val(float[][] x_val) {
        this.x_val = x_val;
    }

    public float[][] getY_oneHot_train() {
        return y_oneHot_train;
    }

    public void setY_oneHot_train(float[][] y_oneHot_train) {
        this.y_oneHot_train = y_oneHot_train;
    }

    public float[][] getY_oneHot_val() {
        return y_oneHot_val;
    }

    public void setY_oneHot_val(float[][] y_oneHot_val) {
        this.y_oneHot_val = y_oneHot_val;
    }

    public float[] getY_train() {
        return y_train;
    }

    public void setY_train(float[] y_train) {
        this.y_train = y_train;
    }

    public float[] getY_val() {
        return y_val;
    }

    public void setY_val(float[] y_val) {
        this.y_val = y_val;
    }

    private float[][] y_oneHot_val;
    private float[] y_train;
    private float[] y_val;

    private TrainableVariable trainableVariable;
    private HashMap<String, float[][]> weight = new HashMap<>();
    private HashMap<String, float[]> bias = new HashMap<>();
    private HashMap<String, String> nameMap = new HashMap<>();

    public TrainInitialize(LocalCSVReader localCSVReader) {
        this.x_train = localCSVReader.getX_train();
        this.y_oneHot_train = localCSVReader.getY_oneHot_train();
        this.y_train = localCSVReader.getY_train();
        this.x_val = localCSVReader.getX_val();
        this.y_oneHot_val = localCSVReader.getY_oneHot_val();
        this.y_val = localCSVReader.getY_val();
    }

    @Override
    public void initVar() {

    }

    /**
     * @param trainableVar     trainable variable of layers
     * @param trainableVarName namespace of lists
     * @param listShape
     */
    public TrainableVariable initVar(List<List<Float>> trainableVar, List<String> trainableVarName,
                                     List<String> targetVarName, List<String> assignVarName,
                                     List<List<Integer>> listShape) {
        TrainableVariable trainableVariable = new TrainableVariable();
        Iterator trainableVarIterator = trainableVar.iterator();
        Iterator trainableVarNameIterator = trainableVarName.iterator();
        Iterator targetVarNameIterator = targetVarName.iterator();
        Iterator listShapeIterator = listShape.iterator();
        while (trainableVarIterator.hasNext() && trainableVarNameIterator.hasNext() &&
                listShapeIterator.hasNext() && targetVarNameIterator.hasNext()) {
            List<Float> floatList = (List<Float>) trainableVarIterator.next();
            String string = (String) trainableVarNameIterator.next();
            String targetString = (String) targetVarNameIterator.next();
            List<Integer> integerList = (List<Integer>) listShapeIterator.next();
            if (integerList.size() == 1) {
                this.bias.put(string, listConvert.floatConvert(floatList, integerList.get(0)));
            } else {
                this.weight.put(string, listConvert.floatConvert2D(floatList,
                        integerList.get(0), integerList.get(1)));

            }
            nameMap.put(string, targetString);

        }
        trainableVariable.setWeight(this.weight);
        trainableVariable.setBias(this.bias);
        trainableVariable.setNameMap(nameMap);
        trainableVariable.setBackPropagationList(assignVarName);
        return trainableVariable;
    }


}
