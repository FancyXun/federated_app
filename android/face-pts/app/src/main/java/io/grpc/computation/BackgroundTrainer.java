package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;

import org.tensorflow.Operation;
import org.tensorflow.Session;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.computation.pipeline.RunOneRound;
import io.grpc.learning.computation.Certificate;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
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
                
                new RunOneRound(session, activityReference.get(), this.context).runOneRound(stub, builder);
                StaticTrainerInfo.ClientInfo.round += 1;
                System.out.println(StaticTrainerInfo.ClientInfo.round);
            }
            return "Training Finished!";
        }
    }
}
