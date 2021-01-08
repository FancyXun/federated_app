package io.grpc.tflite.detect.PCNutil;

import io.grpc.tflite.detect.Model.Pcn1;
import io.grpc.tflite.detect.Model.Pcn2;
import io.grpc.tflite.detect.Model.Pcn3;
import io.grpc.tflite.detect.Model.PcnModel;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

//import java.awt.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class FaceDetect {

    public static void main(String[] args) {

    }

    public static ArrayList<Mat> detect(Mat img, Mat imgPad, ArrayList<PcnModel> pcnModels){
        // Import Models
        Pcn1 pcn1 = (Pcn1) pcnModels.get(0);
        Pcn2 pcn2 = (Pcn2) pcnModels.get(1);
        Pcn3 pcn3 = (Pcn3) pcnModels.get(2);

        Mat img180 = ImageUtil.imageFlipping(imgPad);
        Mat img90 = ImageUtil.imageTranspose(imgPad);
        Mat imgNeg90 = ImageUtil.imageFlipping(img90);

        ArrayList<Window2> winlst1 = PCN.stage1(img, imgPad, 0.37, pcn1);
        ArrayList<Window2> winlst1_nms = PCN.NMS(winlst1, true, 0.8);
        ArrayList<Window2> winlst2 = PCN.stage2(imgPad, img180, 0.43, 24, winlst1_nms, pcn2);
        ArrayList<Window2> winlst2_nms = PCN.NMS(winlst2, true, 0.8);
        ArrayList<Window2> winlst3 = PCN.stage3(imgPad, img180, img90, imgNeg90,0.97, 48, winlst2_nms, pcn3);
        ArrayList<Window2> winlst3_nms = PCN.NMS(winlst3, false, 0.3);
        ArrayList<Window2> winlst_fp = deleteFP(winlst3_nms);
        ArrayList<Window1> faces = PCN.transWindow(img, imgPad, winlst_fp);

        ArrayList<Mat> ret = new ArrayList<>();
        for (Window1 face : faces) {
            Mat cropedFace = ImageUtil.cropImage(img, face, 200);
            ret.add(cropedFace);
        }
        // test
//        Window1 win = faces.get(0);
//        Mat testRet = img.submat(win.getY(),win.getY()+win.getWidth(), win.getX(),win.getX()+win.getWidth());
//        ret.add(testRet);

        return ret;
    }

    public static boolean inside(double x, double y, Window2 rect){
        return rect.getX() <= x && x < (rect.getX() + rect.getW()) && rect.getY() <= y &&
                y < (rect.getY() + rect.getH());
    }

    public static ArrayList<Window2> deleteFP(ArrayList<Window2> winlst){
        int length = winlst.size();
        if(length == 0)
            return winlst;
        Collections.sort(winlst, new Comparator<Window2>() {
            @Override
            public int compare(Window2 o1, Window2 o2) {
                return new Double(o2.getConf()).compareTo(new Double(o1.getConf()));
            }
        });

        int[] flag = new int[length];
        for (int i=0; i<length; i++){
            flag[i] = 0;
        }

        for (int i = 0; i<length; i++){
            if (flag[i] == 0)
                continue;
            for(int j=i+1; j<length; j++){
                Window2 win = winlst.get(j);
                if(inside(win.getX(), win.getY(), winlst.get(i)) &&
                inside(win.getX()+win.getW()-1, win.getY()+win.getH()-1, winlst.get(i))){
                    flag[j] = 1;
                }
            }
        }
        ArrayList<Window2> ret = new ArrayList<>();
        for(int i= 0; i<flag.length; i++){
            if(flag[i] == 0){
                ret.add(winlst.get(i));
            }
        }
    return ret;
    }
}
