package io.grpc.learning.vo;

import java.util.List;

public class SequenceData {
    private List<List<Float>> tensorVar;
    private List<String> tensorName;
    private List<String> tensorTargetName;
    private List<List<Integer>> tensorShape;

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
