package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import org.tensorflow.Session;

import java.lang.ref.WeakReference;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.computation.pipeline.RunOneRound;
import io.grpc.computation.pipeline.Training;
import io.grpc.computation.pipeline.TrainingInit;
import io.grpc.computation.pipeline.WeightsFeed;
import io.grpc.computation.pipeline.WeightsUpload;
import io.grpc.learning.computation.Certificate;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.TrainMetrics;
import io.grpc.transmit.StreamCall;
import io.grpc.utils.Timer;
import io.grpc.vo.StaticTrainerInfo;

public class BackgroundTrainer {
    static class LocalTraining extends StreamCall {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;

        public Context getContext() {
            return context;
        }

        public Session getSession() {
            return session;
        }

        @SuppressLint("StaticFieldLeak")
        private Context context;
        private Session session;
        

        
        protected LocalTraining(Activity activity, Context context) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
        }

        @RequiresApi(api = Build.VERSION_CODES.N)
        @Override
        protected String doInBackground(String... params) {
            
            channel = ManagedChannelBuilder
                    .forAddress(StaticTrainerInfo.ServeInfo.server_ip, StaticTrainerInfo.ServeInfo.server_port)
                    .maxInboundMessageSize(1024 * 1024 * 1024)
                    .usePlaintext().build();
            ComputationGrpc.ComputationBlockingStub stub = ComputationGrpc.newBlockingStub(channel);
            
            while (StaticTrainerInfo.ClientInfo.local_loss > StaticTrainerInfo.ClientInfo.loss_threshold) {
                ClientRequest.Builder builder = ClientRequest.newBuilder().setId(
                        StaticTrainerInfo.ClientInfo.localId);
                
                Certificate certificate = stub.callTraining(builder.build());
                if (StaticTrainerInfo.ClientInfo.token == null) {
                    StaticTrainerInfo.ClientInfo.token = certificate.getToken();
                } else {
                    while (StaticTrainerInfo.ClientInfo.token.equals(certificate.getToken())) {
                        Timer.sleep(3000);
                        certificate = stub.callTraining(builder.build());
                    }
                }
                while (!certificate.getServerState().equals("ready")) {
                    Timer.sleep(3000);
                    certificate = stub.callTraining(builder.build());
                }
                StaticTrainerInfo.ClientInfo.token = certificate.getToken();
                
//                new RunOneRound(session, activityReference.get(), this.context).runOneRound(stub, builder);
                runOneRound(stub, builder);
                StaticTrainerInfo.ClientInfo.round += 1;
                System.out.println(StaticTrainerInfo.ClientInfo.round);
            }
            return "Training Finished!";
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
}
