package io.grpc.learning.api;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;

import javax.imageio.ImageIO;

public class ResNetTest {

    @Test
    public void readPB() {
        Graph graph = new Graph();
        InputStream modelStream = null;
        String path = "/home/zhangxun/data/cifar10_png/train/";
        String[] LabelList = new String[]{
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
        };
        String var2 = "resource/modelMeta/resnet.pb";
        int train_batch = 300;
        int val_batch = 100;
        float[][][][] batch_data = new float[train_batch + val_batch][32][32][3];
        int[] label = new int[train_batch + val_batch];
        float[][][][] train_batch_data = new float[train_batch][32][32][3];
        int[] train_label = new int[train_batch];
        float[][][][] val_batch_data = new float[val_batch][32][32][3];
        int[] val_label = new int[val_batch];

        try {
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
            System.out.println(op);
        }
        System.out.println("----------------------------");
        // train data
        ArrayList<String[]> arrayList = new ArrayList();
        for (int i = 0; i < LabelList.length; i++) {
            File file_list = new File(path + LabelList[i]);
            File[] fs_list = file_list.listFiles();
            for (File f : fs_list) {
                if (!f.isDirectory()) {
                    arrayList.add(new String[]{f.getPath(), String.valueOf(i)});
                }
            }
        }
        Collections.shuffle(arrayList);

        // val data
        session.runner().addTarget("init").run();

        LinkedHashMap linkedHashMap = readTXT();
        try {
            int imgIdx = 0;
            for (String[] f : arrayList) {
                BufferedImage image = ImageIO.read(new FileInputStream(f[0]));
                int height = image.getHeight();
                int width = image.getWidth();
                int channel = 3;
                float sum = 0;
                double std = 0;
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        Color temp = new Color(image.getRGB(i, j));
                        sum = sum + temp.getRed() + temp.getGreen() + temp.getBlue();
                    }
                }
                float average = sum / (width * height * channel);
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        Color temp = new Color(image.getRGB(i, j));
                        std = std + Math.pow(temp.getRed() - average, 2);
                        std = std + Math.pow(temp.getGreen() - average, 2);
                        std = std + Math.pow(temp.getBlue() - average, 2);
                    }
                }
                std = Math.sqrt(std / (width * height * channel - 1));
                std = Math.max(std, Math.sqrt(1.0 / (width * height * 3)));
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        Color temp = new Color(image.getRGB(i, j));
                        batch_data[imgIdx][i][j][0] = (float) ((temp.getRed() - average) / std);
                        batch_data[imgIdx][i][j][1] = (float) ((temp.getGreen() - average) / std);
                        batch_data[imgIdx][i][j][2] = (float) ((temp.getBlue() - average) / std);
                    }
                }
                label[imgIdx] = Integer.valueOf(f[1]);
                // split train an val
                if (imgIdx > 0 && (imgIdx + 1) % (train_batch + val_batch) == 0) {
                    for (int i = 0; i < train_batch; i++) {
                        train_batch_data[i] = batch_data[i];
                        train_label[i] = label[i];
                    }
                    for (int i = 0; i < val_batch; i++) {
                        val_batch_data[i] = batch_data[train_batch + i];
                        val_label[i] = label[train_batch + i];
                    }
                    Session.Runner runner = session.runner()
                            .feed("Placeholder", Tensor.create(0.1f))
                            .feed("Placeholder_1", Tensor.create(val_label))
                            .feed("Placeholder_2", Tensor.create(val_batch_data))
                            .feed("Placeholder_3", Tensor.create(train_label))
                            .feed("Placeholder_4", Tensor.create(train_batch_data));

                    // val
                    runner.addTarget("group_deps").run();
                    Tensor validation_error_value = runner.fetch("truediv_1").run().get(0);
                    Tensor validation_loss_value = runner.fetch("cross_entropy_1").run().get(0);

                    // train bp
                    runner.addTarget("Momentum").run();
                    runner.addTarget("ExponentialMovingAverage").run();
                    Tensor train_loss_value = runner.fetch("AddN").run().get(0);
                    Tensor train_error_value = runner.fetch("truediv").run().get(0);
                    System.out.println("Train top1 error = " + train_error_value.floatValue());
                    System.out.println("Train loss = " + train_loss_value.floatValue());
                    System.out.println("Validation top1 error = " + validation_error_value.floatValue());
                    System.out.println("Validation loss = " + validation_loss_value.floatValue());
                    System.out.println("----------------------------");
                    float [][][][] floats = new float[3][3][3][16];
                    float [][][][] floats1 = new float[3][3][3][16];
                    float [][][][] floats2 = new float[3][3][3][16];
                    for (Object key: linkedHashMap.keySet()){
                        Tensor var = session.runner().fetch(String.valueOf(key)).run().get(0);
                        String shape = (String) linkedHashMap.get(key);
                        System.out.println(var +" "+ shape);
                        var.copyTo(floats);
                        session.runner().feed(String.valueOf(key), Tensor.create(floats1))
                                .addTarget(String.valueOf(key) +"/Assign")
                                .run();
                        Tensor var1 = session.runner().fetch(String.valueOf(key)).run().get(0);
                        var1.copyTo(floats2);
                    }
                    imgIdx = 0;
                } else {
                    imgIdx += 1;
                }

            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public LinkedHashMap<String, String> readTXT() {
        FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null;
        LinkedHashMap<String, String> hashMap = new LinkedHashMap<>();
        try {
            String str = "";
            String str1 = "";
            fis = new FileInputStream("resource/modelMeta/resnet_trainable_var.txt");
            isr = new InputStreamReader(fis);
            br = new BufferedReader(isr);
            while ((str = br.readLine()) != null) {
                str1 += str + "\n";
                String[] strings = str.split(":");
                hashMap.put(strings[0], strings[1]);
            }
//            StringBuilder stringBuilder = new StringBuilder(str1);

            System.out.println(str1);// 打印出str1
        } catch (FileNotFoundException e) {
            System.out.println("找不到指定文件");
        } catch (IOException e) {
            System.out.println("读取文件失败");
        } finally {
            try {
                br.close();
                isr.close();
                fis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return hashMap;
    }
}
