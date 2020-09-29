package io.grpc.utils;

import android.content.Context;

import com.google.common.util.concurrent.ExecutionError;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class LocalCSVReader {
    /**
     *
     */
    private float[][] floatArray;
    // x of training data, x_train + x_val = x
    private float[][] x;
    private float[][] x_train;
    private float[][] x_val;

    // y of training data, y_train + y_val = y
    private float[] y;
    private float[] y_train;
    private float[] y_val;

    // y_oneHot of training data, y_oneHot_train + y_oneHot_val = y_oneHot
    private float[][] y_oneHot;
    private float[][] y_oneHot_train;
    private float[][] y_oneHot_val;

    private String target;
    private int yIndex;
    private String dataSplit;
    String pattern = "(.*@)(([0-9]+)-([0-9]+))*";
    Pattern r = Pattern.compile(pattern);

    public int getHeight() {
        return height;
    }

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

    public float[][] getX_train() {
        return x_train;
    }

    public void setX_train(float[][] x_train) {
        this.x_train = x_train;
    }

    public float[][] getX_val() {
        return x_val;
    }

    public void setX_val(float[][] x_val) {
        this.x_val = x_val;
    }

    public float[] getY_train() {
        return y_train;
    }

    public void setY_train(float[] y_train) {
        this.y_train = y_train;
    }

    public float[] getY_val() {
        return y_val;
    }

    public void setY_val(float[] y_val) {
        this.y_val = y_val;
    }

    public float[][] getY_oneHot_train() {
        return y_oneHot_train;
    }

    public void setY_oneHot_train(float[][] y_oneHot_train) {
        this.y_oneHot_train = y_oneHot_train;
    }

    public float[][] getY_oneHot_val() {
        return y_oneHot_val;
    }

    public void setY_oneHot_val(float[][] y_oneHot_val) {
        this.y_oneHot_val = y_oneHot_val;
    }

    public float[][] getFloatArray() {
        return floatArray;
    }

    private void setFloatArray(float[][] floatArray) {
        this.floatArray = floatArray;
    }


    /**
     * @param context   Android context for read resource
     * @param CSVPath   training data csv path
     * @param header    csv has header or not, 0 or 1
     * @param target
     * @param dataSplit
     */
    public LocalCSVReader(Context context, String CSVPath, int header,
                          String target, String dataSplit) {
        this.target = target;
        this.dataSplit = dataSplit;
        List<List<String>> records = new ArrayList<List<String>>();
        String var = CSVPath;
        boolean var1 = var.startsWith("file:///android_asset/");
        String var2 = var1 ? var.split("file:///android_asset/")[1] : var;
        try (CSVReader csvReader = new CSVReader(new InputStreamReader(context.getAssets().open(var2)))) {
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
        this.train_test_split();
    }

    public void train_test_split() {
        Matcher m = r.matcher(dataSplit);
        if (m.find( )){
            int start = Integer.parseInt(m.group(3));
            int end = Integer.parseInt(m.group(4));
            int trainSize = this.height * (end - start) / 10;
            this.x_train = Arrays.copyOfRange(this.x, 0, trainSize);
            this.y = Arrays.copyOfRange(this.y, 0, trainSize);
            this.y_oneHot_train = Arrays.copyOfRange(this.y_oneHot, 0, trainSize);
            this.x_val = Arrays.copyOfRange(this.x, trainSize, this.height);
            this.y_val = Arrays.copyOfRange(this.y, trainSize, this.height);
            this.y_oneHot_val = Arrays.copyOfRange(this.y_oneHot, trainSize, this.height);
        }
        else{
            System.out.println("No Match!");
        }
    }
}
