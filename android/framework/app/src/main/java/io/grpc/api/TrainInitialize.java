package io.grpc.api;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import io.grpc.utils.LocalCSVReader;
import io.grpc.utils.listConvert;
import io.grpc.vo.TrainableVariable;

public class TrainInitialize implements GraphInitialize {
    private float[][] x;
    private float[][] y_oneHot;

    public float[][] getX() {
        return x;
    }

    public void setX(float[][] x) {
        this.x = x;
    }

    public float[][] getY_oneHot() {
        return y_oneHot;
    }

    public void setY_oneHot(float[][] y_oneHot) {
        this.y_oneHot = y_oneHot;
    }

    public float[] getY() {
        return y;
    }

    public void setY(float[] y) {
        this.y = y;
    }

    private float[] y;
    private TrainableVariable trainableVariable;
    private HashMap<String, float[][]> weight = new HashMap<>();
    private HashMap<String, float[]> bias = new HashMap<>();
    private HashMap<String, String> nameMap = new HashMap<>();

    public TrainInitialize(LocalCSVReader localCSVReader) {
        this.x = localCSVReader.getX();
        this.y_oneHot = localCSVReader.getY_oneHot();
        this.y = localCSVReader.getY();
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