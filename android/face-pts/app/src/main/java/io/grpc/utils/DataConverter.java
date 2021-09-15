package io.grpc.utils;

import org.opencv.core.Mat;

public class DataConverter {

    @Deprecated
    public static float[][][][] cvMat_3dArray(Mat mat, int batch_size) {
        float[][][][] result = new float[batch_size][mat.height()][mat.width()][mat.channels()];
        for (int bat = 0; bat< batch_size; bat++) {
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    for (int c = 0; c < mat.channels(); c++) {
                        result[bat][i][j][c] = (float) (mat.get(i, j)[c] - 127.5)/128;
                    }
                }
            }
        }
        return result;
    }

    public static void cvMat_batchArray(Mat mat, int batch_size_iter, float[][][][] result) {
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    for (int c = 0; c < mat.channels(); c++) {
                        result[batch_size_iter][i][j][c] = (float) (mat.get(i, j)[c] - 127.5)/128;
                    }
                }
            }
    }
}
