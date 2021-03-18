package io.grpc.computation.pipeline;

import android.app.Activity;
import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

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
    @RequiresApi(api = Build.VERSION_CODES.N)
    public void runOneRound(ComputationGrpc.ComputationBlockingStub stub,
                            ClientRequest.Builder builder
                            ) {

        // get model graph and graph info
        this.session = TrainingInit.getInstance().ModelGraphInit(stub, builder);
        // get model weights and feed to session
        WeightsFeed.getInstance().weightsFeed(this.session, stub);
        // training
        Training.getInstance().localTrain(context, session);
        // upload weights to server
        WeightsUpload.getInstance().streamUpload(stub, this.session);

        TrainMetrics.Builder trainBuilder =  TrainMetrics.newBuilder();
        trainBuilder.setId(StaticTrainerInfo.ClientInfo.localId);
        trainBuilder.setAcc((float) StaticTrainerInfo.TrainInfo.acc);
        trainBuilder.setLoss((float) StaticTrainerInfo.TrainInfo.loss);
        trainBuilder.setDataNum(StaticTrainerInfo.TrainInfo.dataNum);
        stub.computeMetrics(trainBuilder.build());
        stub.computeFinish(builder.build());
        this.session.close();
    }
    
}
