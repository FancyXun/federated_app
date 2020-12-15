package io.grpc.computation;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;

import org.tensorflow.Session;

import java.lang.ref.WeakReference;
import java.util.List;
import java.util.UUID;

import io.grpc.ManagedChannel;
import io.grpc.learning.computation.Layer;
import io.grpc.transmit.StreamCall;

public class FrozenTrainer {
    static class LocalTraining extends StreamCall {
        private final WeakReference<Activity> activityReference;
        private ManagedChannel channel;
        @SuppressLint("StaticFieldLeak")
        private Context context;
        private String initName;
        private String optimizerName;
        private String lossName;
        private String dataSplit;
        private Session session;
        private final int maxFloatNumber = 1000000;
        private String server_ip = "192.168.50.38";
        private int server_port = 50051;
        private final String path = "http://192.168.89.154:8888/images"; // image url
        private final String image_txt = "images.txt"; //train images
        private static String localId = UUID.randomUUID().toString().replaceAll("-", "");
        private List<Layer> layerList;
        private int round;


        protected LocalTraining(Activity activity, Context context) {
            this.activityReference = new WeakReference<Activity>(activity);
            this.context = context;
        }

        @Override
        protected String doInBackground(String... params) {
            runOneRound();
            return "training finished !";
        }

        public void runOneRound(){

        }
    }
}
