package io.grpc.transmit;

import android.os.AsyncTask;

import java.util.List;

import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.ComputationRequest;
import io.grpc.learning.computation.TensorValue;
import io.grpc.learning.computation.TrainableVarName;
import io.grpc.learning.computation.ValueReply;
import io.grpc.vo.Metrics;
import io.grpc.vo.SequenceType;

@Deprecated
public class StreamCall extends AsyncTask<String, Void, String> {

    protected int offSet = 0;
    protected SequenceType sequenceType;

    @Deprecated
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

    @Deprecated
    public boolean upload(ComputationGrpc.ComputationBlockingStub stub, String localId,
                          String modelName, List<List<Float>> lists, Metrics metrics) {
        boolean uploaded = false;
        io.grpc.learning.computation.Metrics.Builder metricsBuilder =
                io.grpc.learning.computation.Metrics.newBuilder();
        metricsBuilder.addAllName(metrics.metricsName);
        metricsBuilder.addAllValue(metrics.metrics);
        metricsBuilder.setWeights(metrics.weights);
        for (int i = 0; i < lists.size(); i++) {
            TensorValue.Builder tensorValueBuilder = TensorValue.newBuilder()
                    .setId(localId).setNodeName(modelName);
            tensorValueBuilder.setOffset(i)
                    .setValueSize(lists.size());
            tensorValueBuilder.addAllListArray(lists.get(i));
            if (i == lists.size() - 1) {
                tensorValueBuilder.setMetrics(metricsBuilder);
            }
            ValueReply valueReply = stub.compute(tensorValueBuilder.build());
            uploaded = valueReply.getMessage();
        }
        return uploaded;
    }

    @Override
    protected String doInBackground(String... strings) {
        return null;
    }

}
