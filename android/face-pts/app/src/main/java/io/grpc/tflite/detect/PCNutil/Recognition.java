package io.grpc.tflite.detect.PCNutil;


import android.app.Activity;
import android.util.Log;

import io.grpc.tflite.detect.Model.PcnModel;

import org.opencv.core.Mat;

import java.io.IOException;
import java.util.ArrayList;

public class Recognition {

    public static PcnModel pcnModel;
    public static PcnModel.Device device = PcnModel.Device.CPU;
    public static ArrayList<PcnModel> pcnModels = new ArrayList<>();

    public static void recreateModels(Activity act, PcnModel.Device device, int numThreads) {
        if (pcnModel != null) {
            Log.d("TFlite", "Closing model.");
            pcnModel.close();
            pcnModel = null;
        }
        try {
            Log.d("TFlite", String.format("Creating model (device=%s, numThreads=%d)", device, numThreads));
//            pcnModel = PcnModel.create(this, device, numThreads,0);

            PcnModel model1 = PcnModel.create(act, device, numThreads, 1);
            PcnModel model2 = PcnModel.create(act, device, numThreads, 2);
            PcnModel model3 = PcnModel.create(act, device, numThreads, 3);

            pcnModels.add(model1);
            pcnModels.add(model2);
            pcnModels.add(model3);
        } catch (IOException e) {
            Log.e("TFlite", "Failed to create model.", e);
        }
    }

    public static ArrayList<Mat> RecongFunc(Activity act, Mat src){
        recreateModels(act, device, -1);
        Mat imgPad = ImageUtil.imagePadding(src);
        ArrayList<Mat> ret = FaceDetect.detect(src, imgPad, pcnModels);
        return ret;
    }

}
