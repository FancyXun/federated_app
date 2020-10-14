package io.grpc.learning.api;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import javax.imageio.ImageIO;

public class ResNetTest {

    @Test
    public void readPB() {
        Graph graph = new Graph();
        InputStream modelStream = null;
        String path = "/home/zhangxun/data/cifar10_png/train/airplane";
        try {
            String var2 = "resource/pb/resnet.pb";
            modelStream = new FileInputStream(var2);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] byteArray = buffer.toByteArray();
            graph.importGraphDef(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
        Iterator<Operation> operationIterator = graph.operations();
        while (operationIterator.hasNext()) {
            Operation op = operationIterator.next();
//            System.out.println(op);
        }
        int train_batch = 128;
        int val_batch = 250;
        Float[][][][] train_batch_data = new Float[train_batch][32][32][3];
        Integer[] train_label = new Integer[train_batch];
        Float[][][][] val_batch_data = new Float[val_batch][32][32][3];
        Integer[] val_label = new Integer[val_batch];

        try {
            File file = new File(path);
            File[] fs = file.listFiles();
            for (File f : fs) {
                if (!f.isDirectory()) {
                    BufferedImage image = ImageIO.read(new FileInputStream(f));
                    int height = image.getHeight();
                    int width = image.getWidth();
                    int channel = 3;
                    for (int i = 0; i < width; i++) {
                        for (int j = 0; j < height; j++) {
                            Color temp = new Color(image.getRGB(i, j));
                            train_batch_data[0][i][j][0] = Float.valueOf(temp.getRed());
                            train_batch_data[0][i][j][1] = Float.valueOf(temp.getRed());
                            train_batch_data[0][i][j][2] = Float.valueOf(temp.getRed());
                        }
                    }

                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        session.runner().addTarget("init").run();
        Session.Runner runner = session.runner().feed("Placeholder", Tensor.create(1.0f))
                .feed("Placeholder_1", Tensor.create(val_label))
                .feed("Placeholder_2", Tensor.create(val_batch_data))
                .feed("Placeholder_3", Tensor.create(train_label))
                .feed("Placeholder_4", Tensor.create(train_batch_data));
        runner.addTarget("Momentum").run();
        runner.addTarget("ExponentialMovingAverage").run();
        Tensor res = runner.fetch("AddN").run().get(0);
        Tensor res1 = runner.fetch("truediv").run().get(0);
        System.out.println(res.floatValue());
        System.out.println(res1.floatValue());
    }
}
