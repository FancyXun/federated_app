package io.grpc.vo;

import java.util.List;
import java.util.UUID;

import io.grpc.learning.computation.Layer;

public class StaticTrainerInfo {
    public static class MetaInfo {
        public static String initName;
        public static String x;
        public static String y;
        public static String optimizerName;
        public static String lossName;
        public static String accName;
        public static Boolean oneHot=false;
        public static List<Layer> TrainableLayerList;
    }

    public static class ServeInfo {
        public static String server_ip = "192.168.51.15";
        public static int server_port = 50051;
        public static final String path = "http://52.81.162.253:8000/res/CASIA-WebFace-aligned";
        public static final String image_txt = "images/client0.txt";
    }

    public static class ClientInfo {
        public static String localId =
                UUID.randomUUID().toString().replaceAll("-", "");
        public static int round = 0;
        public static String token = null;
        public static float local_loss = Float.MAX_VALUE;
        public static float loss_threshold = 0.01f;
        public static boolean firstRound = true;
    }

    public static class TrainInfo {
        public static int batch_size = 1;
        public static double loss = 0;
        public static double acc = 0;
        public static int dataNum = 0;
    }
}
