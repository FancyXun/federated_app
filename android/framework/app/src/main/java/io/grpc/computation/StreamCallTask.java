package io.grpc.computation;

import android.os.AsyncTask;

import java.util.List;

import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.learning.computation.TensorValue;
import io.grpc.learning.computation.TrainableVarName;
import io.grpc.learning.computation.ValueReply;
import io.grpc.vo.SequenceType;

public class StreamCallTask extends AsyncTask<String, Void, String> {

    protected int offSet = 0;
    protected SequenceType sequenceType;

    public SequenceType SequenceCall(ComputationGrpc.ComputationBlockingStub stub, ComputationRequest.Builder builder) {
        SequenceType sequenceType = new SequenceType();
        TensorValue tensorValue = stub.callValue(builder.setOffset(offSet).build());
        int size = tensorValue.getValueSize();
        TrainableVarName trainableVarName = tensorValue.getTrainableName();
        sequenceType.getTensorVar().add(tensorValue.getListArrayList());
        sequenceType.getTensorName().add(trainableVarName.getName());
        sequenceType.getTensorTargetName().add(trainableVarName.getTargetName());
        sequenceType.getTensorShape().add(tensorValue.getShapeArrayList());
        sequenceType.getTensorAssignShape().add(tensorValue.getAssignShapeArrayList());
        sequenceType.setTensorAssignName(tensorValue.getAssignNameList());
        sequenceType.setPlaceholder(tensorValue.getPlaceholderList());
        size -= 1;
        while (size > 0) {
            offSet += 1;
            tensorValue = stub.callValue(builder.setOffset(offSet).build());
            trainableVarName = tensorValue.getTrainableName();
            sequenceType.getTensorVar().add(tensorValue.getListArrayList());
            sequenceType.getTensorName().add(trainableVarName.getName());
            sequenceType.getTensorTargetName().add(trainableVarName.getTargetName());
            sequenceType.getTensorShape().add(tensorValue.getShapeArrayList());
            sequenceType.getTensorAssignShape().add(tensorValue.getAssignShapeArrayList());
            size -= 1;
        }
        offSet = 0;
        return sequenceType;
    }

    public boolean upload(ComputationGrpc.ComputationBlockingStub stub, String localId,
                          String modelName, List<List<Float>> lists){
        boolean uploaded = false;
        for (int i =0 ; i< lists.size(); i++){
            System.out.println("*******"+lists.size() + lists.get(i).size());
            TensorValue.Builder tensorValueBuilder = TensorValue.newBuilder()
                    .setId(localId).setNodeName(modelName);
            tensorValueBuilder.setOffset(i)
                    .setValueSize(lists.size());
            tensorValueBuilder.addAllListArray(lists.get(i));
            ValueReply valueReply = stub.sendValue(tensorValueBuilder.build());
            uploaded = valueReply.getMessage();
        }
        return uploaded;
    }

    @Override
    protected String doInBackground(String... strings) {
        return null;
    }
}
