package io.grpc.tflite.detect.Model;

import android.app.Activity;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public abstract class PcnModel {
//    private static final Logger LOGGER = new Logger();

    /**
     * The runtime device type used for executing classification.
     */
    public enum Device {
        CPU,
        GPU
    }

    /**
     * Number of results to show in the UI.
     */
    private static final int MAX_RESULTS = 3;

    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

//  /** Image size along the x axis. */
//  private final int imageSizeX;
//
//  /** Image size along the y axis. */
//  private final int imageSizeY;

    /** Optional GPU delegate for acceleration. */
    // TODO: Declare a GPU delegate


    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    // TODO: Declare a TFLite interpreter
    protected Interpreter tflite;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> labels;

    /**
     * Input image TensorBuffer.
     */
    private TensorImage inputImageBuffer;

    /**
     * Output probability TensorBuffer.
     */
//    protected TensorBuffer outputProbabilityBuffer;

    /**
     * Processer to apply post processing of the output probability.
     */
//    private final TensorProcessor probabilityProcessor;

    /**
     * Creates a classifier with the provided configuration.
     *
     * @param activity   The current Activity.
     * @param device     The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     */
    public static PcnModel create(Activity activity, Device device, int numThreads, int modelIndex)
            throws IOException {
        if (modelIndex == 1) {
            return new Pcn1(activity, device, numThreads);
        } else if (modelIndex == 2) {
            return new Pcn2(activity, device, numThreads);
        } else {
            return new Pcn3(activity, device, numThreads);
        }
    }

    protected PcnModel(Activity activity, Device device, int numThreads) throws IOException {
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());/**引入模型*/
//        tfliteModel = FileUtil.loadMappedFile(activity, "pcn1.tflite");

        switch (device) {
            case GPU:
                // TODO: Create a GPU delegate instance and add it to the interpreter options

                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);

        // TODO: Create a TFLite interpreter instance
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Loads labels out from the label file (useless
//        labels = FileUtil.loadLabels(activity, getLabelPath());
//        System.out.println(labels);/**add*/

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}

        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        System.out.println(imageDataType);
        System.out.println(Arrays.toString(imageShape));
        int probabilityTensorIndex = 0;
//    int[] probabilityShape =
//        tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
//        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);
        System.out.println(inputImageBuffer);
        // Creates the output tensor and its processor.
//    outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
//    outputProbabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 2}, probabilityDataType);
        //outputProbabilityBuffer = TensorBuffer.createFixedSize(new int[]{1,2}, probabilityDataType);
        // Creates the post processor for the output probability.
//        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        Log.d("TFlite","Created a Tensorflow Lite Image Model.");
    }

    /** Gets the name of the model file stored in Assets. */
    protected abstract String getModelPath();

//    /** Gets the name of the label file stored in Assets. */
//    protected abstract String getLabelPath();
    public void close() {
        if (tflite != null) {
            // TODO: Close the interpreter
            tflite.close();
            tflite = null;
        }
        // TODO: Close the GPU delegate


        tfliteModel = null;
    }

}
