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

    public void setTensorAssignName(List<String> tensorAssignName) {
        this.tensorAssignName = tensorAssignName;
    }

    public List<String> getPlaceholder() {
        return placeholder;
    }

    public void setPlaceholder(List<String> placeholder) {
        this.placeholder = placeholder;
    }

    private List<List<Integer>> tensorShape = new ArrayList<>();

    public List<List<Float>> getTensorVar() {
        return tensorVar;
    }

    public void setTensorVar(List<List<Float>> tensorVar) {
        this.tensorVar = tensorVar;
    }

    public List<String> getTensorName() {
        return tensorName;
    }

    public void setTensorName(List<String> tensorName) {
        this.tensorName = tensorName;
    }

    public List<String> getTensorTargetName() {
        return tensorTargetName;
    }

    public void setTensorTargetName(List<String> tensorTargetName) {
        this.tensorTargetName = tensorTargetName;
    }

    public List<List<Integer>> getTensorShape() {
        return tensorShape;
    }

    public void setTensorShape(List<List<Integer>> tensorShape) {
        this.tensorShape = tensorShape;
    }


}
