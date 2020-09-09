package io.grpc.learning.utils;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class LocalCSVReader {
    /**
     *
     */
    private float[][] floatArray;
    private float[][] x;
    private float[] y;
    private float[][] y_oneHot;
    private String target;
    private int yIndex;
    private int height, width;

    public float[][] getX() {
        return x;
    }

    public void setX(float[][] x) {
        this.x = x;
    }


    public float[] getY() {
        return y;
    }

    private void setY(float[] y) {
        this.y = y;
    }

    public float[][] getY_oneHot() {
        return y_oneHot;
    }

    private void setY_oneHot(float[][] y_oneHot) {
        this.y_oneHot = y_oneHot;
    }


    public float[][] getFloatArray() {
        return floatArray;
    }

    private void setFloatArray(float[][] floatArray) {
        this.floatArray = floatArray;
    }


    /**
     * @param CSVPath training data csv path
     * @param header  csv has header or not, 0 or 1
     */
    public LocalCSVReader(String CSVPath, int header, String target) {
        this.target = target;
        List<List<String>> records = new ArrayList<List<String>>();
        try (CSVReader csvReader = new CSVReader(new FileReader(CSVPath));) {
            String[] values;
            if (header == 0) {
                String[] valuesHeader = csvReader.readNext();
                List valuesHeaderList = Arrays.asList(valuesHeader);
                this.yIndex = valuesHeaderList.indexOf(target);
            }
            while ((values = csvReader.readNext()) != null) {
                records.add(Arrays.asList(values));
            }
            String[][] array = new String[records.size()][];
            for (int i = 0; i < records.size(); i++) {
                List<String> row = records.get(i);
                array[i] = row.toArray(new String[row.size()]);
            }
            float[][] floatArray = new float[array.length][array[0].length];
            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    floatArray[i][j] = Float.parseFloat(array[i][j]);
                }
            }
            this.setFloatArray(floatArray);
            this.height = floatArray.length;
            this.width = floatArray[0].length;
            this.splitLabel();
        } catch (IOException | CsvValidationException e) {
            e.printStackTrace();
        }
    }

    private void splitLabel() {
        float[][] x = new float[this.height][this.width - 1];
        float[] y = new float[this.height];
        for (int i = 0; i < this.floatArray.length; i++) {
            for (int j = 0; j < this.floatArray[0].length; j++) {
                if (j == this.yIndex) {
                    y[i] = this.floatArray[i][j];
                } else {
                    x[i][j > this.yIndex ? j - 1 : j] = this.floatArray[i][j];
                }
            }
        }
        this.x = x;
        this.y = y;
        this.oneHot();
    }

    private void oneHot() {
        HashSet<Float> set = new HashSet<Float>();
        for (Float f : this.y) {
            set.add(f);
        }
        HashMap<Float, Integer> toIndex = new HashMap<>();
        int ind = 0;
        for (Float f : set) {
            toIndex.put(f, ind);
            ind++;
        }
        float[][] y_oneHot = new float[this.height][set.size()];
        for (int i = 0; i < this.y.length; i++) {
            float[] a = new float[set.size()];
            a[toIndex.get(this.y[i])] = 1;
            y_oneHot[i] = a;
        }
        this.y_oneHot = y_oneHot;
    }
}
