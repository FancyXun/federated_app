package io.grpc.computation;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Timestamp;
import java.util.ArrayList;

import io.grpc.utils.FileUtils;

import static io.grpc.utils.ImageTools.l2Normalize;
import static io.grpc.utils.ImageTools.loadModelFile;
import static io.grpc.utils.ImageTools.normalizeImage;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private Button trainButton;
    private Button faceUpload;
    private Button faceRec;
    private ImageView faceImg;
    private Bitmap bitmap;
    private Context context;
    private ArrayList<String> fileList;
    private Classifier classifier;
    private TextView textView;
    private static String liteModelUrl = "http://52.81.162.253:8000/res/model_train.tflite";
    private static String localLiteModelUrl = "/data/user/0/io.grpc.computation/cache/model";
    private CascadeClassifier cascadeClassifier;
    private File mCascadeFile;

    private Interpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        faceUpload = findViewById(R.id.faceUpload);
//        faceImg = findViewById(R.id.faceImg);
        faceRec = findViewById(R.id.faceRec);
//        textView = findViewById(R.id.Similarity);
        trainButton = (Button) findViewById(R.id.train);
        context = getApplicationContext();
        fileList = new FileUtils(context, "inference").getFileList();
        loadHaarCascadeFile();
//        faceUpload.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Intent intent = new Intent();
//                intent.setType("image/*");
//                intent.setAction(Intent.ACTION_GET_CONTENT);
//                /*
//                Starting another activity, whether one within your app or from another app,
//                doesn't need to be a one-way operation. You can also start another activity
//                and receive a result back.
//                 */
//                startActivityForResult(intent, 1);
//            }
//        });

    }

    private void loadHaarCascadeFile() {
        try {
            File cascadeDir = getDir("haarcascade_frontalface_alt", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");

            if (!mCascadeFile.exists()) {
                FileOutputStream os = new FileOutputStream(mCascadeFile);
                InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }
        } catch (Throwable throwable) {
            throw new RuntimeException("Failed to load Haar Cascade file");
        }
    }


    public void Training(View view) {
        trainButton.setEnabled(false);
//        new TrainerStream.LocalTraining(this, this.context).execute(
//                "NULL"
//        );
//        new FrozenTrainer.LocalTraining(this, this.context).execute(
//                "NULL"
//        );

        new BackgroundTrainer.LocalTraining(this, this.context).execute();

    }

    public void inference_local(View view) throws IOException {
        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        cascadeClassifier.load(mCascadeFile.getAbsolutePath());
        faceRec.setEnabled(false);
        classifier = Classifier.create(this, Classifier.Device.CPU, 1);
        MatOfRect matOfRect = new MatOfRect();
        Bitmap bmp_detect = null;
        float[][] emb = new float[1][];
        for (String filePath : fileList) {
            if (filePath.contains("test")){
                Mat image = Imgcodecs.imread(cacheFile(filePath).getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
                Bitmap bmp = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(image, bmp);
                cascadeClassifier.detectMultiScale(image, matOfRect);
                Rect[] facesArray = matOfRect.toArray();
                for (Rect rect : facesArray) {
                    Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0),2);
                }
                try {
                    bmp_detect = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(image, bmp_detect);
                }
                catch (CvException e){
                    Log.d("Exception", e.getMessage());}
                emb = mobileFaceNetinference(bmp_detect);
            }
        }

        for (String filePath : fileList) {
            if (filePath.contains("test")){
                continue;
            }
            Mat image = Imgcodecs.imread(cacheFile(filePath).getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
            Bitmap bmp = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image, bmp);
            float[][] emb1 = mobileFaceNetinference(bmp);
            double similarity = cosineSimilarity(emb[0], emb1[0]);
            System.out.println(filePath+ "similarity:" + similarity);
        }
        faceRec.setEnabled(true);
    }

    public void inference(View view) throws IOException {
        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        cascadeClassifier.load(mCascadeFile.getAbsolutePath());
        faceRec.setEnabled(false);
//        TFLiteFileUtil.downloadFile(liteModelUrl, new File(localLiteModelUrl));
        classifier = Classifier.create(this, Classifier.Device.CPU, 1);
        Bitmap bit = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        Mat src = new Mat(bit.getHeight(), bit.getWidth(), CvType.CV_8UC(3));
        Utils.bitmapToMat(bit, src);
        MatOfRect matOfRect = new MatOfRect();
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("检测" + timestamp);
        cascadeClassifier.detectMultiScale(src, matOfRect);
        Rect[] facesArray = matOfRect.toArray();
        for (Rect rect : facesArray) {
            Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0),2);
        }
        Bitmap bmp_detect = null;
        try {
            bmp_detect = Bitmap.createBitmap(src.cols(), src.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(src, bmp_detect);
        }
        catch (CvException e){
            Log.d("Exception", e.getMessage());}

        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("检测" + timestamp);
        float[] results =
                classifier.recognizeImage(bmp_detect, 90);
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("识别" + timestamp);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        StringBuilder res = new StringBuilder();
        for (String filePath : fileList) {
            Mat image = Imgcodecs.imread(cacheFile(filePath).getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
            Bitmap bmp = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image, bmp);
            float[] results1 =
                    classifier.recognizeImage(bmp, 90);
            double similarity = cosineSimilarity(results, results1);
            System.out.println("similarity：" + similarity);
            res.append(similarity).append(";");
            compare(bmp, bmp_detect);
            mobileFaceNetinference(bmp_detect);
            localPbTest(bmp_detect);
        }
        textView.setText(res.toString());
        faceRec.setEnabled(true);
    }

    public float[][] mobileFaceNetinference(Bitmap bitmap){
        String MODEL_FILE = "rec/mobileFaceNet.tflite";
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        try {
            interpreter = new Interpreter(loadModelFile(getAssets(), MODEL_FILE), options);
        } catch (IOException e) {
            e.printStackTrace();
        }

        int INPUT_IMAGE_SIZE = 112;
        Bitmap bitmapScale = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        float[][][][] datasets = new float[1][INPUT_IMAGE_SIZE][INPUT_IMAGE_SIZE][3];
        datasets[0] = normalizeImage(bitmapScale);
        float[][] embeddings = new float[1][192];
        interpreter.run(datasets, embeddings);
        l2Normalize(embeddings, 1e-10);
        return embeddings;
    }

    public void localPbTest(Bitmap bitmap){
        Graph graph = new Graph();
        InputStream modelStream = null;
        String pbPath = "protobuffer/mobileFaceNet.pb";
        int INPUT_IMAGE_SIZE = 112;
        Bitmap bitmapScale = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        float[][][][] datasets = new float[1][INPUT_IMAGE_SIZE][INPUT_IMAGE_SIZE][3];
        datasets[0] = normalizeImage(bitmapScale);
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
        session.runner().addTarget("init").run();
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("识别pb" + timestamp);
        session.runner().feed("input", Tensor.create(datasets)).fetch("embeddings").run();
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("识别pb" + timestamp);
        System.out.println("end..." );

    }

    public float compare(Bitmap bitmap1, Bitmap bitmap2) {
        // 将人脸resize为112X112大小的，因为下面需要feed数据的placeholder的形状是(2, 112, 112, 3)
        String MODEL_FILE = "rec/MobileFaceNet2Img.tflite";
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        try {
            interpreter = new Interpreter(loadModelFile(getAssets(), MODEL_FILE), options);
        } catch (IOException e) {
            e.printStackTrace();
        }

        int INPUT_IMAGE_SIZE = 112;
        Bitmap bitmapScale1 = Bitmap.createScaledBitmap(bitmap1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        Bitmap bitmapScale2 = Bitmap.createScaledBitmap(bitmap2, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);

        float[][][][] datasets = getTwoImageDatasets(bitmapScale1, bitmapScale2);
        float[][] embeddings = new float[2][192];
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("识别1" + timestamp);
        interpreter.run(datasets, embeddings);
        l2Normalize(embeddings, 1e-10);
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("识别1" + timestamp);
        float same = evaluate(embeddings);
        System.out.println("识别1..." + same);
        return same;
    }

    private float[][][][] getTwoImageDatasets(Bitmap bitmap1, Bitmap bitmap2) {
        Bitmap[] bitmaps = {bitmap1, bitmap2};
        int INPUT_IMAGE_SIZE = 112;
        int[] ddims = {bitmaps.length, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3};
        float[][][][] datasets = new float[ddims[0]][ddims[1]][ddims[2]][ddims[3]];

        for (int i = 0; i < ddims[0]; i++) {
            Bitmap bitmap = bitmaps[i];
            datasets[i] = normalizeImage(bitmap);
        }
        return datasets;
    }

    private float evaluate(float[][] embeddings) {
        float[] embeddings1 = embeddings[0];
        float[] embeddings2 = embeddings[1];
        float dist = 0;
        for (int i = 0; i < 192; i++) {
            dist += Math.pow(embeddings1[i] - embeddings2[i], 2);
        }
        float same = 0;
        for (int i = 0; i < 400; i++) {
            float threshold = 0.01f * (i + 1);
            if (dist < threshold) {
                same += 1.0 / 400;
            }
        }
        return same;
    }

    private File cacheFile(String filename) {
        File file = new File(context.getCacheDir() + "/tmp");
        try {
            InputStream is = context.getAssets().open(filename);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(buffer);
            fos.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return file;
    }

    public static double cosineSimilarity(float[] vectorA, float[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            Uri uri = data.getData();
            assert uri != null;
            Log.e("uri", uri.toString());
            ContentResolver cr = this.getContentResolver();
            try {
                //Get image
                bitmap = BitmapFactory.decodeStream(cr.openInputStream(uri));
                faceImg.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                Log.e("Exception", e.getMessage(), e);
            }
        } else {
            Log.i("MainActivity", "operation error or no image to choose");
        }
        super.onActivityResult(requestCode, resultCode, data);
    }


    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.i("cv", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
        } else {
            Log.i("cv", "OpenCV library found inside package. Using it!");
        }
    }

    @SuppressLint("HandlerLeak")
    Handler handler = new Handler() {

        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            if (msg.what == 1) {
                faceImg.setImageBitmap(bitmap);
            }
        }
    };
}
