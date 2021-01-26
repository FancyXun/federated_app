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
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Timestamp;
import java.util.ArrayList;

import io.grpc.tflite.detect.PCNutil.ImageUtil;
import io.grpc.tflite.detect.PCNutil.Recognition;
import io.grpc.utils.FileUtils;
import io.grpc.utils.ImageUtils;
import io.grpc.utils.TFLiteFileUtil;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        faceUpload = findViewById(R.id.faceUpload);
        faceImg = findViewById(R.id.faceImg);
        faceRec = findViewById(R.id.faceRec);
        textView = findViewById(R.id.Similarity);
        trainButton = (Button) findViewById(R.id.train);
        context = getApplicationContext();
        fileList = new FileUtils(context, "sampleData/casiaWebFace").getFileList();
        faceUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                /*
                Starting another activity, whether one within your app or from another app,
                doesn't need to be a one-way operation. You can also start another activity
                and receive a result back.
                 */
                startActivityForResult(intent, 1);
            }
        });

    }


    public void Training(View view) {
        trainButton.setEnabled(false);
//        new TrainerStream.LocalTraining(this, this.context).execute(
//                "NULL"
//        );
        new FrozenTrainer.LocalTraining(this, this.context).execute(
                "NULL"
        );

    }

    public void inference(View view) throws IOException {
        faceRec.setEnabled(false);
//        TFLiteFileUtil.downloadFile(liteModelUrl, new File(localLiteModelUrl));
        classifier = Classifier.create(this, Classifier.Device.CPU, 1);
        Bitmap bit = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        Mat src = new Mat(bit.getHeight(), bit.getWidth(), CvType.CV_8UC(3));
//        Utils.bitmapToMat(bit, recognition.RecongFunc(this, src).get(0));
        Utils.bitmapToMat(bit, src);
        Recognition recognition = new Recognition();

        Size size = new Size(255,255);
        Imgproc.resize(src, src, size);
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("检测" + timestamp);
        System.out.println(recognition.RecongFunc(this, ImageUtil.fourC2threeC(src)));
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println("检测" + timestamp);
        float[] results =
                classifier.recognizeImage(bit, 90);
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
        }
        textView.setText(res.toString());
        faceRec.setEnabled(true);
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
