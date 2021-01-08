package io.grpc.tflite.detect.PCNutil;

import io.grpc.tflite.detect.Model.Pcn1;
import io.grpc.tflite.detect.Model.Pcn2;
import io.grpc.tflite.detect.Model.Pcn3;
import io.grpc.tflite.detect.Model.PcnModel;

import org.opencv.core.Mat;

//import javax.print.DocFlavor;
//import java.awt.*;
import java.lang.reflect.Array;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;

public class PCN {
    public static final int STRIDE = 8;
    public static final double SCALE_ = 1.414;
    public static final double EPS = 1e-5;
    public static final double ANGLERANGE = 45.;

    public static double IOU(Window2 w1, Window2 w2){
        double xOverLap = Math.max(0,Math.min(w1.getX()+w1.getW()-1,
                w2.getX()+w2.getW()-1)-Math.max(w1.getX(),w2.getX())+1);
        double yOverLap = Math.max(0,Math.min(w1.getY()+w1.getH()-1,
                w2.getY()+w2.getH()-1)-Math.max(w1.getY(),w2.getH())+1);
        double intersection = xOverLap*yOverLap;
        double uion = w1.getW() * w1.getH() + w2.getW()*w2.getH() - intersection;

        return intersection/uion;
    }

    public static ArrayList<Window2> NMS(ArrayList<Window2> winlst, boolean local, double threshold){
        int length = winlst.size();
        if (length == 0){
            return winlst;
        }
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

        for (int i=0; i<length; i++){
            if (flag[i] == 0){
                continue;
            }
            for(int j=i+1; j<length; j++){
                double wi = winlst.get(i).getScale();
                double wj = winlst.get(j).getScale();
                if(local && Math.abs(wi-wj)>1e-5){
                    continue;
                }
                double iou_val = IOU(winlst.get(i), winlst.get(j));
                if(iou_val>threshold){
                    flag[j] = 1;
                }
            }

        }
        ArrayList<Window2> ret = new ArrayList<>();
        for (int i = 0; i<length; i++){
            if (flag[i] != 1){
                ret.add(winlst.get(i));
            }
        }
        return ret;
    }

    private static boolean legal(double x, double y, Mat img){
        if (0<=x && x<img.width() && 0<=y && y<img.height())
            return true;
        else
            return false;
    }

    public static ArrayList<Window1> transWindow(Mat img, Mat imgPad, ArrayList<Window2> winlst){
        int row = (imgPad.height() - img.height())/2;
        int col = (imgPad.width() - img.width())/2;
        ArrayList<Window1> ret = new ArrayList<>();
        for (Window2 window2 : winlst) {
            if (window2.getW()>0 && window2.getH()>0) {
                int x = window2.getX() - col;
                int y = window2.getY() - row;
                ret.add(new Window1(x, y, window2.getW(), window2.getAngle(), window2.getConf()));
            }
        }
        return ret;
    }

    public static ArrayList<Window2> stage1(Mat img, Mat imgPad, double threshold, Pcn1 stage1Model){
        ArrayList<Window2> winlst = new ArrayList<>();
        int row = (imgPad.height() - img.height())/2;
        int col = (imgPad.width() - img.width())/2;
        int netSize = 24;
        double curScale = (double)28/24;

        Mat img_resized = ImageUtil.imageResize(img, curScale);

        // 可以删掉
//        double[][][][] cls_prob = new double[1][][][];
//        double[][][][] rotate = new double[1][][][];
//        double[][][][] bbox = new double[1][][][];
        while(Math.min(img_resized.width(), img_resized.height()) >= netSize){
            img_resized = ImageUtil.preprocessImage(img_resized,0);
            int imgArraySize = (img_resized.width()-24)/8 + 1;
            float[][][][] imgArray = ImageUtil.imgClipModel1(img_resized, imgArraySize);
            /*
            TODO: 模型训练
                获取3个输出：cls_prob[][][][], rotate[][][][], bbox[][][][]
                shape(1,x,y,2)   (1,x,y,2)   (1,x,y,3)
            double[][][][] imgArray = ImageUtil.imgClipModel1(img_resized);
            // Pcn1 stage1Model: tflite model
            // stage1: inference function
            Map<Integer, Object> output = stage1Model.stage1(imgArray, imgArray.length);
            double[][][][] cls_prob = (double[][][][]) output.get(0);
            double[][][][] rotate = (double[][][][]) output.get(1);
            double[][][][] bbox = (double[][][][]) output.get(2);
             */
            Map<Integer, Object> output = stage1Model.stage1(imgArray, imgArray.length);
            float[][][][] cls_prob = (float[][][][]) output.get(0);
            float[][][][] rotate = (float[][][][]) output.get(1);
            float[][][][] bbox = (float[][][][]) output.get(2);

            double w = netSize * curScale;
//            for (int i = 0; i<imgArray[0].length; i++){
//                for (int j = 0; j<imgArray[0][0].length; j++){
//                    if (cls_prob[i*imgArray[0].length+j][0][0][0]>threshold){
//                        double sn = bbox[i*imgArray[0].length+j][0][0][0];
//                        double xn = bbox[i*imgArray[0].length+j][0][0][1];
//                        double yn = bbox[i*imgArray[0].length+j][0][0][2];
//                        int rx = (int)(j*curScale*STRIDE - 0.5*sn*w + sn*xn*w + 0.5*w) + col;
//                        int ry = (int)(i*curScale*STRIDE - 0.5*sn*w + sn*yn*w + 0.5*w) + row;
//                        int rw = (int)(w*sn);
//                        if (legal(rx, ry, imgPad) && legal(rx+rw-1, ry+rw-1, imgPad)){
//                            if (rotate[i*imgArray[0].length+j][0][0][1] > 0.5)
//                                winlst.add(new Window2(rx, ry, rw, rw, 0, curScale, cls_prob[i*imgArray[0].length+j][0][0][1]));
//                            else
//                                winlst.add(new Window2(rx, ry, rw, rw, 180, curScale, cls_prob[i*imgArray[0].length+j][0][0][1]));
//                        }
//
//                    }
//                }
//            }
            for (int i = 0; i<imgArray.length; i++){
                if (cls_prob[i][0][0][1]>threshold){
                    double sn = bbox[i][0][0][0];
                    double xn = bbox[i][0][0][1];
                    double yn = bbox[i][0][0][2];
                    int idx_i = i/imgArraySize;
                    int idx_j = i%imgArraySize;
                    int rx = (int)(idx_j*curScale*STRIDE - 0.5*sn*w + sn*xn*w + 0.5*w) + col;
                    int ry = (int)(idx_i*curScale*STRIDE - 0.5*sn*w + sn*yn*w + 0.5*w) + row;
                    int rw = (int)(w*sn);
                    if (legal(rx, ry, imgPad) && legal(rx+rw-1, ry+rw-1, imgPad)){
                        if (rotate[i][0][0][1] > 0.5)
                            winlst.add(new Window2(rx, ry, rw, rw, 0, curScale, cls_prob[i][0][0][1]));
                        else
                            winlst.add(new Window2(rx, ry, rw, rw, 180, curScale, cls_prob[i][0][0][1]));
                    }
                }
            }
            img_resized = ImageUtil.imageResize(img_resized, SCALE_);
            curScale = (double)img.height() / img_resized.height();
        }
        return winlst;
    }

    public static ArrayList<Window2> stage2(Mat imgPad, Mat img180, double threshold, int dim,
                                            ArrayList<Window2> winlst, Pcn2 stage2Model){
        int length = winlst.size();
        if (length == 0)
            return winlst;
        ArrayList<Mat> datalst = new ArrayList<>();
        int height = imgPad.height();
        for (int i = 0; i<winlst.size(); i++){
            Window2 win = winlst.get(i);
            if (Math.abs(winlst.get(i).getAngle()) < EPS){
                Mat pro_img = ImageUtil.preprocessImage(imgPad.submat(win.getY(),win.getY()+win.getH(), win.getX(),win.getX()+win.getW()),dim);
                datalst.add(pro_img);
            }
            else{
                int y2 = win.getY() + win.getH() - 1;
                int y = height - 1 - y2;
                Mat pro_img = ImageUtil.preprocessImage(img180.submat(y,y+win.getH(),win.getX(), win.getX()+win.getW()), dim);
                datalst.add(pro_img);
            }
        }
        float[][][][] imgArray = new float[length][][][];
        for (int i =0; i<length; i++){
            imgArray[i] = ImageUtil.mat2Array(datalst.get(i))[0];
        }

        /*
        TODO: 模型训练
            获取3个输出：cls_prob[][], rotate[][], bbox[][]
            shape(x,2)   (x,2)   (x,3)
        // Pcn2 stage2Model: tflite model
        // stage2: inference function
        Map<Integer, Object> output = stage2Model.stage2(imgArray, imgArray.length);
        double[][] cls_prob = (double[][]) output.get(0);
        double[][] rotate = (double[][]) output.get(1);
        double[][] bbox = (double[][]) output.get(2);
         */
        ArrayList<Window2> new_winlst = new ArrayList<>();
        Map<Integer, Object> output = stage2Model.stage2(imgArray, imgArray.length);
        float[][] cls_prob = (float[][]) output.get(0);
        float[][] rotate = (float[][]) output.get(1);
        float[][] bbox = (float[][]) output.get(2);

        //可以删掉
//        double[][] cls_prob = new double[length][2];
//        double[][] rotate = new double[length][2];
//        double[][] bbox = new double[length][3];

        for (int i = 0; i <length; i++){
            Window2 win = winlst.get(i);
            if (cls_prob[i][1] > threshold){
                double sn = bbox[i][0];
                double xn = bbox[i][1];
                double yn = bbox[i][2];
                int cropX = win.getX();
                int cropY = win.getY();
                int cropW = win.getW();
                if (Math.abs(win.getAngle()) > EPS){
                    cropY = height - 1 - (cropY + cropW -1);
                }
                int w = (int)(sn * cropW);
                int x = (int)(cropX - 0.5*sn*cropW + cropW*sn*xn + 0.5*cropW);
                int y = (int)(cropY - 0.5*sn*cropW + cropW*sn*yn + 0.5*cropW);
                double maxRotateScore = 0;
                double maxRotateIndex = 0;
                for (int j=0; j<3; j++){
                    if (rotate[i][j] > maxRotateScore){
                        maxRotateScore = rotate[i][j];
                        maxRotateIndex = j;
                    }
                }
                if (legal(x, y, imgPad) && legal(x+w-1, y+w-1, imgPad)){
                    double angle = 0;
                    if (Math.abs(win.getAngle()) < EPS){
                        if (maxRotateIndex == 0)
                            angle = 90;
                        else if (maxRotateIndex == 1)
                            angle = 0;
                        else
                            angle = -90;
                        new_winlst.add(new Window2(x, y, w, w, angle, win.getScale(), cls_prob[i][1]));
                    }
                    else{
                        if (maxRotateIndex == 0)
                            angle = 90;
                        else if (maxRotateIndex == 1)
                            angle = 180;
                        else
                            angle = -90;
                        new_winlst.add(new Window2(x, height-1-(y+w-1), w, w, angle, win.getScale(), cls_prob[i][1]));
                    }
                }
            }
        }
        return new_winlst;
    }

    public static ArrayList<Window2> stage3(Mat imgPad, Mat img180, Mat img90, Mat imgNeg90,
                                            double threshold, int dim, ArrayList<Window2> winlst,
                                            Pcn3 stage3Model){
        int length = winlst.size();
        if (length == 0)
            return winlst;

        ArrayList<Mat> datalst = new ArrayList<>();
        int height = imgPad.height();
        int width = imgPad.width();
        for (int i = 0; i<length; i++){
            Window2 win = winlst.get(i);
            if (Math.abs(win.getAngle())< EPS){
                Mat subimg = imgPad.submat(win.getY(), win.getY()+win.getH(), win.getX(), win.getX()+win.getW());
                Mat pro_img = ImageUtil.preprocessImage(subimg, dim);
                datalst.add(pro_img);
            }
            else if (Math.abs(win.getAngle() - 90) < EPS){
                Mat subimg = img90.submat(win.getX(), win.getX()+win.getW(), win.getY(), win.getY()+win.getH());
                Mat pro_img = ImageUtil.preprocessImage(subimg, dim);
                datalst.add(pro_img);
            }
            else if (Math.abs(win.getAngle() + 90) < EPS){
                int x = win.getY();
                int y = width - 1 - (win.getX() + win.getW() - 1);
                Mat subimg = imgNeg90.submat(y, y+win.getH(), x, x+win.getW());
                Mat pro_img = ImageUtil.preprocessImage(subimg, dim);
                datalst.add(pro_img);
            }
            else{
                int y2 = win.getY() + win.getH() - 1;
                int y = height - 1 - y2;
                Mat subimg = img180.submat(y, y+win.getH(), win.getX(), win.getX()+win.getW());
                Mat pro_img = ImageUtil.preprocessImage(subimg, dim);
                datalst.add(pro_img);
            }
        }

        float[][][][] imgArray = new float[length][][][];
        for (int i =0; i<length; i++){
            imgArray[i] = ImageUtil.mat2Array(datalst.get(i))[0];
        }

        /*
        TODO: 模型训练
            获取3个输出：cls_prob[][], rotate[][], bbox[][]
            shape:       (x,2)         (x,1)      (x,3)
        // Pcn3 stage3Model: tflite model
        // stage3: inference function
        Map<Integer, Object> output = stage3Model.stage3(imgArray, imgArray.length);
        double[][] cls_prob = (double[][]) output.get(0);
        double[][] rotate = (double[][]) output.get(1);
        double[][] bbox = (double[][]) output.get(2);
         */

        ArrayList<Window2> new_winlst = new ArrayList<>();
        Map<Integer, Object> output = stage3Model.stage3(imgArray, imgArray.length);
        float[][] cls_prob = (float[][]) output.get(0);
        float[][] rotate = (float[][]) output.get(1);
        float[][] bbox = (float[][]) output.get(2);
        //可以删掉
//        double[][] cls_prob = new double[length][2];
//        double[][] rotate = new double[length][1];
//        double[][] bbox = new double[length][3];

        for (int i = 0; i<length; i++){
            Window2 win = winlst.get(i);
            if (cls_prob[i][1] > threshold){
                double sn = bbox[i][0];
                double xn = bbox[i][1];
                double yn = bbox[i][2];
                int cropX = win.getX();
                int cropY = win.getY();
                int cropW = win.getW();
                Mat img_tmp = imgPad;
                if (Math.abs(win.getAngle() - 180) < EPS){
                    cropY = height - 1 - (cropY + cropW - 1);
                    img_tmp = img180;
                }
                else if (Math.abs(win.getAngle() - 90) < EPS){
                    int tmp_value = cropX;
                    cropX = cropY;
                    cropY = tmp_value;
                    img_tmp = img90;
                }
                else if (Math.abs(win.getAngle() + 90) < EPS){
                    cropX = win.getY();
                    cropY = width - 1 - (win.getX() + win.getW() - 1);
                    img_tmp = imgNeg90;
                }
                int w = (int)(sn*cropW);
                int x = (int)(cropX - 0.5*sn*cropW + cropW*sn*xn + 0.5*cropW);
                int y = (int)(cropY - 0.5*sn*cropW + cropW*sn*yn + 0.5*cropW);
                double angle = ANGLERANGE * rotate[i][0];

                if (legal(x,y,img_tmp) && legal(x+w-1, y+w-1, img_tmp)){
                    if (Math.abs(win.getAngle()) < EPS)
                        new_winlst.add(new Window2(x, y, w, w, angle, win.getScale(), cls_prob[i][1]));
                    else if (Math.abs(win.getAngle() - 180) < EPS)
                        new_winlst.add(new Window2(x, height-1-(y+w-1), w, w, 180-angle, win.getScale(), cls_prob[i][1]));
                    else if (Math.abs(win.getAngle() - 90) < EPS)
                        new_winlst.add(new Window2(y, x, w, w, 90-angle, win.getScale(), cls_prob[i][1]));
                    else
                        new_winlst.add(new Window2(width-y-w, x, w, w, angle-90, win.getScale(), cls_prob[i][1]));
                }
            }
        }
        return new_winlst;
    }
}
