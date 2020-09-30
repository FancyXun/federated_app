package io.grpc.api;

import android.content.Context;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class SessionMetrics {

    private static String pbPath = "file:///android_asset/metrics/auc.pb";
    private int numThresholds = 1000;
    public Session session;

    public SessionMetrics(Context context) {
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            boolean var1 = pbPath.startsWith("file:///android_asset/");
            String var2 = var1 ? pbPath.split("file:///android_asset/")[1] : pbPath;
            modelStream = context.getAssets().open(var2);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] byteArray = buffer.toByteArray();
            graph.importGraphDef(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Session session = new Session(graph);
        float[] initVar = new float [numThresholds];
        session.runner().feed("auc_pair/true_positives/Initializer/zeros",
                Tensor.create(initVar)).addTarget("auc_pair/true_positives/Assign").run();
        session.runner().feed("auc_pair/false_positives/Initializer/zeros",
                Tensor.create(initVar)).addTarget("auc_pair/false_positives/Assign").run();
        session.runner().feed("auc_pair/true_negatives/Initializer/zeros",
                Tensor.create(initVar)).addTarget("auc_pair/true_negatives/Assign").run();
        session.runner().feed("auc_pair/false_negatives/Initializer/zeros",
                Tensor.create(initVar)).addTarget("auc_pair/false_negatives/Assign").run();
        this.session = session;
    }

}
