package io.grpc.tflite.detect.PCNutil;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.*;
import java.util.ArrayList;

public class ImageUtil {

    public static final int MIN_VAL = 0;
    public static final int MAX_VAL = 255;

    public static float[][][][] mat2Array(Mat src){
        float[][][][] img = new float[1][src.height()][src.width()][3];
        for (int i =0 ; i<src.height(); i++){
            for(int j=0; j<src.width(); j++){
                img[0][i][j][0] = (float) src.get(i,j)[0];
                img[0][i][j][1] = (float) src.get(i,j)[1];
                img[0][i][j][2] = (float) src.get(i,j)[2];
            }
        }
        return img;
    }

    public static Mat readImgRGB(String path){
        Mat src = Imgcodecs.imread(path);
        Imgproc.cvtColor(src,src,Imgproc.COLOR_BGR2RGB);
        return src;
    }

    public static Mat imagePadding(Mat mat){
        int row, col;
        Mat dst = new Mat();
        row = Math.min((int)(mat.height()*0.2), 100);
        col = Math.min((int)(mat.width()*0.2), 100);
        Core.copyMakeBorder(mat, dst, row, row, col, col, Core.BORDER_CONSTANT);
        return dst;
    }

    public static Mat imageFlipping(Mat mat){
        Mat dst = new Mat();
        Core.flip(mat, dst, 0);
        return dst;
    }

    public static Mat imageTranspose(Mat mat){
        Mat dst = new Mat();
        Core.transpose(mat, dst);
        return dst;
    }

    public static Mat imageResize(Mat mat, double scale){
        Mat dst = new Mat();
        int h = mat.height();
        int w = mat.width();
        int h_ = (int)(h/scale);
        int w_ = (int)(w/scale);
        Imgproc.resize(mat, dst, new Size(w_,h_),0,0, Imgproc.INTER_NEAREST);
        return dst;
    }

    public static Mat preprocessImage(Mat mat, int dim){
        if(dim != 0){
            Imgproc.resize(mat, mat, new Size(dim,dim),0,0, Imgproc.INTER_NEAREST);
        }
//        Mat dst = new Mat(mat.height(), mat.width(), CvType.CV_32SC3);
        Mat dst = new Mat();
        mat.convertTo(dst, CvType.CV_32SC3);
        int[] img_data = new int[mat.width() * mat.height() * mat.channels()];
        dst.get(0,0, img_data);
        for (int i = 0; i<mat.total(); i++){
            img_data[i*3] = img_data[i*3] - 104;
            img_data[i*3+1] = img_data[i*3+1] - 117;
            img_data[i*3+2] = img_data[i*3+2] - 123;
        }
        dst.put(0,0,img_data);
        return dst;
    }
	
	public static Mat fourC2threeC(Mat src){
        Mat tar = new Mat(src.height(), src.width(), CvType.CV_32SC3);
        double[] img_data = new double[src.width() * src.height() * 3];
//        tar.get(0,0,img_data);
        for (int i = 0; i < src.total(); i++){
            img_data[i*3] = src.get(i/src.width(), i%src.width())[2];
            img_data[i*3+1] = src.get(i/src.width(), i%src.width())[1];
            img_data[i*3+2] = src.get(i/src.width(), i%src.width())[0];
        }
        tar.put(0,0,img_data);
        return tar;
    }

    public static Mat threeC2fourC(Mat src){
        Mat tar = new Mat(src.height(), src.width(), CvType.CV_8UC4);
        double[] img_data = new double[src.width() * src.height() * 4];
//        tar.get(0,0,img_data);
        for (int i = 0; i < src.total(); i++){
            img_data[i*4] = src.get(i/src.width(), i%src.width())[2];
            img_data[i*4+1] = src.get(i/src.width(), i%src.width())[1];
            img_data[i*4+2] = src.get(i/src.width(), i%src.width())[0];
            img_data[i*4+3] = 255;
        }
        tar.put(0,0,img_data);
        return tar;
    }

    private static int[] rotatePoint(int x, int y, int centerX, int centerY, double angle){
        x -= centerX;
        y -= centerY;
        double theta = -angle * Math.PI / 180;
        int rx = (int)(centerX + x*Math.cos(theta) - y*Math.sin(theta));
        int ry = (int)(centerY + x*Math.sin(theta) + y*Math.cos(theta));
        return new int[]{rx, ry};
    }

    public static Mat cropImage(Mat img, Window1 face, int cropSize){
        int x1 = face.getX();
        int y1 = face.getY();
        int x2 = face.getWidth() + face.getX() - 1;
        int y2 = face.getWidth() + face.getY() - 1;
        int centerX = (x1 + x2)/2;
        int centerY = (y1 + y2)/2;
        int[][] lst = {{x1,y1}, {x1,y2}, {x2,y2}, {x2,y1}};
        ArrayList<int[]> pointList = new ArrayList<>();
        for (int i = 0; i<4; i++){
            int x = lst[i][0];
            int y = lst[i][1];
            int[] tmplst = rotatePoint(x, y, centerX, centerY, face.getAngle());
            pointList.add(tmplst);
        }
        Point[] srcTriangleP = new Point[3];
        srcTriangleP[0] = new Point(pointList.get(0)[0], pointList.get(0)[1]);
        srcTriangleP[1] = new Point(pointList.get(1)[0], pointList.get(1)[1]);
        srcTriangleP[2] = new Point(pointList.get(2)[0], pointList.get(2)[1]);
        Point[] dstTriangleP = new Point[3];
        dstTriangleP[0] = new Point(0,0);
        dstTriangleP[1] = new Point(0, cropSize - 1);
        dstTriangleP[2] = new Point(cropSize - 1, cropSize - 1);
        MatOfPoint2f srcTriangle = new MatOfPoint2f(srcTriangleP);
        MatOfPoint2f dstTriangle = new MatOfPoint2f(dstTriangleP);
        Mat rotMat = Imgproc.getAffineTransform(srcTriangle, dstTriangle);
        Mat ret = new Mat(img.height(), img.width(), CvType.CV_32SC3);
        Imgproc.warpAffine(img, ret, rotMat, new Size(cropSize, cropSize),0);
        return ret;
    }

    public static float[][][][] imgClipModel1(Mat src, int arraySize){
//        int size = src.width();
//        int arraySize = (size-24)/8;
        float[][][][] imgArrays = new float[arraySize*arraySize][][][];
        for (int i=0; i<arraySize; i++){
            for (int j=0; j<arraySize; j++){
                Mat subimg = src.submat(i*8,i*8+24, j*8,j*8+24);
                imgArrays[i*arraySize+j] = ImageUtil.mat2Array(subimg)[0];
            }
        }
        return imgArrays;
    }
}

