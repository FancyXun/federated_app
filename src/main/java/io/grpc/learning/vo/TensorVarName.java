package io.grpc.learning.vo;

import java.util.ArrayList;
import java.util.List;

public class TensorVarName {
    private List<String> tensorName = new ArrayList<>();

    public List<String> getPlaceholder() {
        return placeholder;
    }

    public void setPlaceholder(List<String> placeholder) {
        this.placeholder = placeholder;
    }

    private List<String> placeholder = new ArrayList<>();
    private List<String> tensorTargetName = new ArrayList<>();
    private List<List<Integer>> tensorShape = new ArrayList<>();

    public List<String> getTensorAssignName() {
        return tensorAssignName;
    }

    public void setTensorAssignName(List<String> tensorAssignName) {
        this.tensorAssignName = tensorAssignName;
    }

    private List<String> tensorAssignName = new ArrayList<>();

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
