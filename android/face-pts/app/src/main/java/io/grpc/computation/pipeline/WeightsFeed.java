package io.grpc.computation.pipeline;

import org.tensorflow.Session;
import org.tensorflow.Tensor;

import io.grpc.computation.TrainerStreamUtils;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.Layer;
import io.grpc.vo.StaticTrainerInfo;

public class WeightsFeed {


    private volatile static WeightsFeed instance = null;

    private WeightsFeed() {

    }

    public static WeightsFeed getInstance() {
        if (instance == null) {
            synchronized (WeightsFeed.class) {
                if (instance == null) {
                    instance = new WeightsFeed();
                }
            }

        }
        return instance;
    }

    
    public void weightsFeed(Session session,
                            ComputationGrpc.ComputationBlockingStub stub){
        for (Layer layer: StaticTrainerInfo.MetaInfo.TrainableLayerList) {
            System.out.println("get weights of " + layer.getLayerName());
            Tensor tensor = TrainerStreamUtils.getLayerWeightsByName(StaticTrainerInfo.ClientInfo.localId,
                    layer.getLayerName(), stub);
            session.runner().feed(layer.getLayerInitName(),
                    tensor).addTarget(layer.getLayerName() + "/Assign").run();
            tensor.close();
        }
    }
    
}
