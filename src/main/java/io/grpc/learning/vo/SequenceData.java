package io.grpc.learning.vo;

import java.util.ArrayList;
import java.util.List;

public class SequenceData {
    private List<List<Float>> tensorVar = new ArrayList<>();
    private List<String> tensorName = new ArrayList<>();
    private List<String> tensorTargetName = new ArrayList<>();
    private List<String> tensorAssignName = new ArrayList<>();
    private List<String> placeholder = new ArrayList<>();

    public List<String> getTensorAssignName() {
        return tensorAssignName;
    }


    public List<String> getPlaceholder() {
        return placeholder;
    }

    public void setPlaceholder(List<String> placeholder) {
        this.placeholder = placeholder;
    }

    private List<List<Integer>> tensorShape = new ArrayList<>();

    public List<List<Integer>> getTensorAssignShape() {
        return tensorAssignShape;
    }


    private List<List<Integer>> tensorAssignShape = new ArrayList<>();

    public List<List<Float>> getTensorVar() {
        return tensorVar;
    }


    public List<String> getTensorName() {
        return tensorName;
    }


    public List<String> getTensorTargetName() {
        return tensorTargetName;
    }


    public List<List<Integer>> getTensorShape() {
        return tensorShape;
    }

    public void assignTensorVar(TensorVarName tensorVarName){
        for (int i = 0; i < tensorVarName.getTensorName().size(); i++) {
            List<Integer> integerList = tensorVarName.getTensorShape().get(i);
            List<Integer> integerList1 = tensorVarName.getTensorAssignShape().get(i);
            this.getTensorName().add(tensorVarName.getTensorName().get(i));
            this.getTensorAssignName().add(tensorVarName.getTensorAssignName().get(i));
            this.getTensorTargetName().add(tensorVarName.getTensorTargetName().get(i));
            this.getTensorShape().add(integerList);
            this.getTensorAssignShape().add(integerList1);
            int varSize = 0;
            if (integerList.size() == 1) {
                varSize = integerList.get(0);
            }
            if (integerList.size() == 2) {
                varSize = integerList.get(0) * integerList.get(1);
            }
            List<Float> var = new ArrayList<>(varSize);
            for (int ii = 0; ii < varSize; ii++) {
                var.add(0f);
            }
            this.getTensorVar().add(var);
        }
        for (int i = 0; i < tensorVarName.getPlaceholder().size(); i++) {
            this.getPlaceholder().add(tensorVarName.getPlaceholder().get(i));
        }
    }



}
