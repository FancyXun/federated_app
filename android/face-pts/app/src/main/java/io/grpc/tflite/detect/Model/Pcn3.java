package io.grpc.tflite.detect.Model;

import android.app.Activity;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Pcn3 extends PcnModel {

    /**
     * Float MobileNet requires additional normalization of the used input.
     */
    private static final float IMAGE_MEAN = 0f;

    private static final float IMAGE_STD = 255f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code ClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public Pcn3(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    // TODO: Specify model.tflite as the model file and labels.txt as the label file
    @Override
    protected String getModelPath() {
        return "detect/pcn3.tflite";
    }

//    @Override
//    protected String getLabelPath() {
//        return "lab1.txt";
//    }


//    @Override
//    protected TensorOperator getPreprocessNormalizeOp() {
//        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
//    }
//
//    @Override
//    protected TensorOperator getPostprocessNormalizeOp() {
//        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
//    }

    public Map<Integer,Object> stage3(float[][][][] intputArr, int sampleNum) {

//        System.out.println(Arrays.toString(intputArr));
        // Runs the inference call.
        // TODO: Run TFLite inference
//    tflite.run(bf, outputProbabilityBuffer.getBuffer().rewind());
        //单输出

        //多输出
        Object[] inputs = new Object[]{intputArr};
        float[][] outputc = new float[sampleNum][2];
        float[][] outputr = new float[sampleNum][1];
        float[][] outputb = new float[sampleNum][3];
        Map<Integer, Object> moutput = new HashMap<>();
        moutput.put(0, outputc);
        moutput.put(1, outputr);
        moutput.put(2, outputb);
        tflite.runForMultipleInputsOutputs(inputs, moutput);
        //System.out.println("output:"+ Arrays.toString(output));

        // Gets results.
        return moutput;
    }
}