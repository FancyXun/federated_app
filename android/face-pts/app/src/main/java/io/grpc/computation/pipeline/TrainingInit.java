package io.grpc.computation.pipeline;

import org.tensorflow.Graph;
import org.tensorflow.Session;

import java.util.List;

import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.Meta;
import io.grpc.learning.computation.Model;
import io.grpc.vo.StaticTrainerInfo;

public class TrainingInit {

    private volatile static TrainingInit instance = null;

    public Graph getGraph() {
        return graph;
    }

    private Graph graph;

    private TrainingInit() {

    }

    public static TrainingInit getInstance() {
        if (instance == null) {
            synchronized (TrainingInit.class) {
                if (instance == null) {
                    instance = new TrainingInit();
                }
            }

        }
        return instance;
    }
    
    
    public Session ModelGraphInit(ComputationGrpc.ComputationBlockingStub stub,
                                  ClientRequest.Builder builder) {
        // call model
        Model model = stub.callModel(builder.build());
        StaticTrainerInfo.MetaInfo.oneHot = model.getOneHot();
        StaticTrainerInfo.MetaInfo.labelNum = model.getLabelNum();
        StaticTrainerInfo.MetaInfo.dataUrl = model.getDataUrl();
        StaticTrainerInfo.MetaInfo.width = model.getWidth();
        StaticTrainerInfo.MetaInfo.height = model.getHeight();
        StaticTrainerInfo.MetaInfo.TrainableLayerList = model.getLayerList();
        List<Meta> metaList = model.getMetaList();
        StaticTrainerInfo.MetaInfo.y = metaList.get(0).getMetaName();
        StaticTrainerInfo.MetaInfo.x = metaList.get(1).getMetaName();
        String initName = metaList.get(2).getMetaName();
        StaticTrainerInfo.MetaInfo.optimizerName = metaList.get(3).getMetaName();
        StaticTrainerInfo.MetaInfo.lossName = metaList.get(4).getMetaName();
        StaticTrainerInfo.MetaInfo.accName = metaList.get(5).getMetaName();
        graph = new Graph();
        graph.importGraphDef(model.getGraph().toByteArray());
        Session session = new Session(graph);
        session.runner().addTarget(initName).run();
        return session;
    }
}
