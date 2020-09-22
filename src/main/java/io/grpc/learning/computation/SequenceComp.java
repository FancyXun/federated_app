package io.grpc.learning.computation;

import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.TensorVarName;

public class SequenceComp {

    /**
     * Initialize SequenceData from TensorVarName
     *
     * @param tensorVarName
     * @return SequenceData
     */
    public static SequenceData initializerSequence(TensorVarName tensorVarName) {
        SequenceData sequenceData = new SequenceData();
        sequenceData.assignTensorVar(tensorVarName);
        return sequenceData;
    }

    /**
     * @param sequenceData
     * @param offset
     * @return
     */
    public static TensorValue.Builder offsetStreamReply(SequenceData sequenceData, int offset) {
        TensorValue.Builder reply = TensorValue.newBuilder();
        TrainableVarName.Builder trainableVarName = TrainableVarName.newBuilder();
        trainableVarName.setName(sequenceData.getTensorName().get(offset));
        trainableVarName.setTargetName(sequenceData.getTensorTargetName().get(offset));
        reply.setValueSize(sequenceData.getTensorVar().size());
        reply.addAllShapeArray(sequenceData.getTensorShape().get(offset));
        reply.addAllListArray(sequenceData.getTensorVar().get(offset));
        reply.setTrainableName(trainableVarName);
        reply.addAllAssignName(sequenceData.getTensorAssignName());
        reply.addAllPlaceholder(sequenceData.getPlaceholder());
        reply.addAllAssignShapeArray(sequenceData.getTensorAssignShape().get(offset));
        return reply;
    }
}
