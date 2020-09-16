package io.grpc.computation;

import android.os.AsyncTask;

import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.learning.computation.TensorValue;
import io.grpc.learning.computation.TrainableVarName;
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
        size -= 1;
        while (size > 0) {
            offSet += 1;
            tensorValue = stub.callValue(builder.setOffset(offSet).build());
            sequenceType.getTensorVar().add(tensorValue.getListArrayList());
            sequenceType.getTensorName().add(trainableVarName.getName());
            sequenceType.getTensorTargetName().add(trainableVarName.getTargetName());
            sequenceType.getTensorShape().add(tensorValue.getShapeArrayList());
            size -= 1;
        }
        return sequenceType;
    }

    @Override
    protected String doInBackground(String... strings) {
        return null;
    }
}
