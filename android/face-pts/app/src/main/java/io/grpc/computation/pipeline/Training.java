package io.grpc.computation.pipeline;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;

import androidx.annotation.RequiresApi;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import io.grpc.computation.TrainerStreamUtils;
import io.grpc.utils.DataConverter;
import io.grpc.vo.StaticTrainerInfo;
import io.grpc.vo.ImageInfo;

public class Training {

    private volatile static Training instance = null;

    private Training() {

    }

    public static Training getInstance() {
        if (instance == null) {
            synchronized (Training.class) {
                if (instance == null) {
                    instance = new Training();
                }
            }

        }
        return instance;
    }

    @SuppressLint("SetTextI18n")
    public void train(Context context,
                      Session session) {

        ImageInfo imageInfo = new ImageInfo();
        ArrayList<Float> train_loss_list = new ArrayList<>();
        ArrayList<Float> train_acc_list = new ArrayList<>();
        try {
            // todo: get images from assets
            InputStreamReader inputReader = new InputStreamReader(
                    context.getAssets().open(StaticTrainerInfo.ServeInfo.image_txt));
            BufferedReader buffReader = new BufferedReader(inputReader);
            String line;
            int line_number = 0;
            float[][][][] x = new float[StaticTrainerInfo.TrainInfo.batch_size][imageInfo.getHeight()]
                    [imageInfo.getWidth()][imageInfo.getChannel()];
            int batch_size_iter = 0;
            int[][] label_batch_onehot = null;
            int[] label_batch = null;
            if (StaticTrainerInfo.MetaInfo.oneHot) {
                label_batch_onehot = new int[StaticTrainerInfo.TrainInfo.batch_size][imageInfo.getLabel_num()];
            } else {
                label_batch = new int[StaticTrainerInfo.TrainInfo.batch_size];
            }
            while ((line = buffReader.readLine()) != null) {
                try {
                    Mat image = TrainerStreamUtils.getImage(StaticTrainerInfo.ServeInfo.path + line, imageInfo);
                    int label = Integer.parseInt(line.split("/")[1]);

                    if (StaticTrainerInfo.MetaInfo.oneHot) {
                        label_batch_onehot[batch_size_iter][label] = 1;
                    } else {
                        label_batch[batch_size_iter] = label;
                    }

                    assert image != null;
                    DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                } catch (Exception e) {
                    e.printStackTrace();
                    continue;
                }
                if (batch_size_iter < StaticTrainerInfo.TrainInfo.batch_size - 1) {
                    batch_size_iter++;
                    line_number++;
                    continue;
                } else {
                    batch_size_iter = 0;
                    line_number++;
                }

                Session.Runner runner = session.runner();
                Tensor x_t = Tensor.create(x);
                Tensor label_batch_tensor;
                if (StaticTrainerInfo.MetaInfo.oneHot) {
                    label_batch_tensor = Tensor.create(label_batch_onehot);
                } else {
                    label_batch_tensor = Tensor.create(label_batch);
                }
                Tensor lr_t = Tensor.create(0.0001f);
                runner
                        .feed(StaticTrainerInfo.MetaInfo.x, x_t)
                        .feed(StaticTrainerInfo.MetaInfo.y, label_batch_tensor)
                        .addTarget(StaticTrainerInfo.MetaInfo.optimizerName)
                        .run();
                List<Tensor<?>> fetched_tensors = runner
                        .fetch(StaticTrainerInfo.MetaInfo.lossName)
                        .fetch(StaticTrainerInfo.MetaInfo.accName)
                        .run();

                System.out.println("-----" + ": " + line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                        " acc: " + fetched_tensors.get(1).floatValue());
                train_loss_list.add(fetched_tensors.get(0).floatValue());
                train_acc_list.add(fetched_tensors.get(1).floatValue());
                if (StaticTrainerInfo.MetaInfo.oneHot) {
                    label_batch_onehot = new int[StaticTrainerInfo.TrainInfo.batch_size][imageInfo.getLabel_num()];
                } else {
                    label_batch = new int[StaticTrainerInfo.TrainInfo.batch_size];
                }
                x_t.close();
                label_batch_tensor.close();
                lr_t.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    @RequiresApi(api = Build.VERSION_CODES.N)
    public void localTrain(Context context,
                           Session session) {

        ImageInfo imageInfo = new ImageInfo();
        ArrayList<Float> train_loss_list = new ArrayList<>();
        ArrayList<Float> train_acc_list = new ArrayList<>();

        try {
            // todo: get images from assets
            InputStreamReader inputReader = new InputStreamReader(
                    context.getAssets().open(StaticTrainerInfo.ServeInfo.image_txt));
            BufferedReader buffReader = new BufferedReader(inputReader);
            String line;
            int line_number = 0;
            float[][][][] x = new float[StaticTrainerInfo.TrainInfo.batch_size][imageInfo.getHeight()]
                    [imageInfo.getWidth()][imageInfo.getChannel()];
            int batch_size_iter = 0;
            int[][] label_batch_onehot = null;
            int[] label_batch = null;
            if (StaticTrainerInfo.MetaInfo.oneHot) {
                label_batch_onehot = new int[StaticTrainerInfo.TrainInfo.batch_size][imageInfo.getLabel_num()];
            } else {
                label_batch = new int[StaticTrainerInfo.TrainInfo.batch_size];
            }
            while ((line = buffReader.readLine()) != null) {
                try {
                    Mat image = new Mat();
                    InputStream inputStream = context.getAssets().open("images" + line);
                    Bitmap bmp = BitmapFactory.decodeStream(inputStream);
                    Utils.bitmapToMat(bmp, image);
                    Imgproc.cvtColor(image, image, Imgproc.COLOR_RGBA2RGB);
                    Size size = new Size(imageInfo.getWidth(), imageInfo.getHeight());
                    Imgproc.resize(image, image, size);
                    int label = Integer.parseInt(line.split("/")[1]);

                    if (StaticTrainerInfo.MetaInfo.oneHot) {
                        label_batch_onehot[batch_size_iter][label] = 1;
                    } else {
                        label_batch[batch_size_iter] = label;
                    }

                    assert image != null;
                    DataConverter.cvMat_batchArray(image, batch_size_iter, x);
                } catch (Exception e) {
                    e.printStackTrace();
                    continue;
                }
                if (batch_size_iter < StaticTrainerInfo.TrainInfo.batch_size - 1) {
                    batch_size_iter++;
                    line_number++;
                    continue;
                } else {
                    batch_size_iter = 0;
                    line_number++;
                }

                Session.Runner runner = session.runner();
                Tensor x_t = Tensor.create(x);
                Tensor label_batch_tensor;
                if (StaticTrainerInfo.MetaInfo.oneHot) {
                    label_batch_tensor = Tensor.create(label_batch_onehot);
                } else {
                    label_batch_tensor = Tensor.create(label_batch);
                }
                Tensor lr_t = Tensor.create(0.0001f);
                runner
                        .feed(StaticTrainerInfo.MetaInfo.x, x_t)
                        .feed(StaticTrainerInfo.MetaInfo.y, label_batch_tensor)
                        .addTarget(StaticTrainerInfo.MetaInfo.optimizerName)
                        .run();
                List<Tensor<?>> fetched_tensors = runner
                        .fetch(StaticTrainerInfo.MetaInfo.lossName)
                        .fetch(StaticTrainerInfo.MetaInfo.accName)
                        .run();

                System.out.println("-----" + ": " + line_number + " loss: " + fetched_tensors.get(0).floatValue() +
                        " acc: " + fetched_tensors.get(1).floatValue());
                train_loss_list.add(fetched_tensors.get(0).floatValue());
                train_acc_list.add(fetched_tensors.get(1).floatValue());
                if (StaticTrainerInfo.MetaInfo.oneHot) {
                    label_batch_onehot = new int[StaticTrainerInfo.TrainInfo.batch_size][imageInfo.getLabel_num()];
                } else {
                    label_batch = new int[StaticTrainerInfo.TrainInfo.batch_size];
                }
                x_t.close();
                label_batch_tensor.close();
                lr_t.close();
            }

            StaticTrainerInfo.TrainInfo.loss = train_loss_list.stream().mapToDouble(a -> a).average().getAsDouble();
            StaticTrainerInfo.TrainInfo.acc = train_acc_list.stream().mapToDouble(a -> a).average().getAsDouble();
            StaticTrainerInfo.TrainInfo.dataNum = train_acc_list.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
