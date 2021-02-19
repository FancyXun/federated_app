package io.grpc.learning.computation;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Random;

import sun.misc.IOUtils;

public class TrainingMemoryTest {

    public class ImageInfo {
        int batch_size = 16;
        float total_loss = 0;
        int height = 112;
        int width = 96;
        int channel = 3;
        int label_num = 10575;

        public int getBatch_size() {
            return batch_size;
        }

        public float getTotal_loss() {
            return total_loss;
        }

        public int getHeight() {
            return height;
        }

        public int getWidth() {
            return width;
        }

        public int getChannel() {
            return channel;
        }

        public int getLabel_num() {
            return label_num;
        }
    }

    static class TrainInfo {
        public static int batch_size = 64;
        public static float total_loss = 0;
    }

    static class ServeInfo {
        public static String server_ip = "192.168.0.102";
        public static int server_port = 50051;
        public static final String path = "http://52.81.162.253:8000/res/CASIA-WebFace-aligned";
        public static final String image_txt = "images.txt";
    }

    @Test
    public void local_train() {
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            modelStream = new FileInputStream("resource/modelMeta/sphere_frozen_1234_commandline/sphere_unfrozen.pb");
            graph.importGraphDef(IOUtils.readAllBytes(modelStream));
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
        session.runner().addTarget("init").run();
        ImageInfo imageInfo = new ImageInfo();
        // todo: get images from assets
        String line;
        int line_number = 0;
        float[][][][] x = new float[TrainInfo.batch_size][imageInfo.getHeight()]
                [imageInfo.getWidth()][imageInfo.getChannel()];
        int[][] label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
        Random random = new Random();
        for (int epoch = 0; epoch < 200000; epoch++) {
            x[0][0][0][0] = random.nextFloat();
//            try { Thread.sleep ( 30 ) ;
//            } catch (InterruptedException ie){}
            Session.Runner runner = session.runner();
            Tensor x_t = Tensor.create(x);
            Tensor label_oneHot_t = Tensor.create(label_oneHot);
            Tensor lr_t = Tensor.create(0.0001f);
                runner
                        .feed("input_x", x_t)
                        .feed("input_y", label_oneHot_t)
                        .feed("lr:0", lr_t);
                x_t.close();
                label_oneHot_t.close();
                lr_t.close();
//                                .addTarget("Momentum")
//                                .run();
//
//
//                        List<Tensor<?>> fetched_tensors = runner
//                                .fetch("Mean:0")
//                                .fetch("Mean_1:0")
//                                .run();
//
//                        System.out.println("-----" + ": " + line_number + " loss: " + fetched_tensors.get(0).floatValue() +
//                                " acc: " + fetched_tensors.get(1).floatValue());
                label_oneHot = new int[TrainInfo.batch_size][imageInfo.getLabel_num()];
                line_number ++;
                System.out.println("-------"+": " + line_number);
        }

    }
}
