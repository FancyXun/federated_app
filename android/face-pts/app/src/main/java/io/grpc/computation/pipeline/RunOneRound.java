package io.grpc.computation.pipeline;

import android.app.Activity;
import android.content.Context;

import org.tensorflow.Session;

import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.TrainMetrics;
import io.grpc.vo.StaticTrainerInfo;

public class RunOneRound {
    
    public Session session;
    public Activity activity;
    public Context context;

    public RunOneRound(Session session, Activity activity, Context context){
        this.session = session;
        this.activity = activity;
        this.context = context;
    }
    public void runOneRound(ComputationGrpc.ComputationBlockingStub stub,
                            ClientRequest.Builder builder
                            ) {

        // get model graph and graph info
        this.session = new TrainingInit().ModelGraphInit(stub, builder);
        // get model weights and feed to session
        new WeightsFeed().weightsFeed(this.session, stub);
        // training
        new Training().localTrain(context, session);
        // upload weights to server
        new WeightsUpload().streamUpload(stub, this.session);

        TrainMetrics.Builder trainBuilder =  TrainMetrics.newBuilder();
        trainBuilder.setId(StaticTrainerInfo.ClientInfo.localId);
        trainBuilder.setRound(StaticTrainerInfo.ClientInfo.round);
//        stub.computeMetrics(trainBuilder.build());
        stub.computeFinish(builder.build());
        this.session.close();
    }
    
}
