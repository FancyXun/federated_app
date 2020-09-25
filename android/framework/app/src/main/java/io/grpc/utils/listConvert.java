package io.grpc.utils;

import java.util.ArrayList;
import java.util.List;

public class listConvert {
    /**
     * Convert list to 2d float
     *
     * @param floatList the list of float
     * @param w         2d float array width
     * @param h         2d float array height
     * @return floats of 2d float array
     */
    public static float[][] floatConvert2D(List<Float> floatList, int h, int w) {
        float[][] floats = new float[h][w];
        System.out.println(floatList.size() + "," + h + "," + w);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                floats[i][j] = floatList.get(i * w + j);
            }
        }
        return floats;
    }

    public static float[] floatConvert(List<Float> floatList, int listLen) {
        float[] floats = new float[listLen];
        for (int i = 0; i < listLen; i++) {
            floats[i] = floatList.get(i);
        }
        return floats;
    }

    public static List<Float> listConvert2D(float[][] floats) {
        List<Float> floatList = new ArrayList<>();
        for (int i = 0; i < floats.length; i++) {
            for (int j = 0; j < floats[0].length; j++) {
                floatList.add(floats[i][j]);
            }
        }
        return floatList;
    }

    public static List<Float> listConvert(float[] floats) {
        List<Float> floatList = new ArrayList<>();
        for (int i = 0; i < floats.length; i++) {
            floatList.add(floats[i]);
        }
        return floatList;
    }
}
