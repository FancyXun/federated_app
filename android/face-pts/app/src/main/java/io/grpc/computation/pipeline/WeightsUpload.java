package io.grpc.computation.pipeline;

import org.tensorflow.Session;
import org.tensorflow.Tensor;

import io.grpc.computation.TrainerStreamUtils;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.Layer;
import io.grpc.learning.computation.ValueReply;
import io.grpc.vo.StaticTrainerInfo;

public class WeightsUpload {
    private volatile static WeightsUpload instance = null;

    private WeightsUpload() {

    }

    public static WeightsUpload getInstance() {
        if (instance == null) {
            synchronized (WeightsUpload.class) {
                if (instance == null) {
                    instance = new WeightsUpload();
                }
            }

        }
        return instance;
    }
    public void streamUpload(ComputationGrpc.ComputationBlockingStub stub,
                              Session session) {
        ValueReply valueReply = null;
        TrainerStreamUtils trainerStreamUtils = new TrainerStreamUtils();
        for (Layer layer : StaticTrainerInfo.MetaInfo.TrainableLayerList) {
            Tensor weights = session.runner().
                    fetch(layer.getLayerName()).run().get(0);
            ClientRequest.Builder clientRequestBuilder = ClientRequest.newBuilder();
            clientRequestBuilder.setToken(StaticTrainerInfo.ClientInfo.token);
            clientRequestBuilder.setId(StaticTrainerInfo.ClientInfo.localId);
            valueReply = trainerStreamUtils.callLayerWeights(clientRequestBuilder,
                    layer.getLayerName(), stub, weights,
                    layer.getLayerShape());
            System.out.println(layer.getLayerName());
            weights.close();
        }
    }
}
