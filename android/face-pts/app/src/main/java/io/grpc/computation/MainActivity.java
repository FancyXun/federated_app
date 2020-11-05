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

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Tensor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;

import io.grpc.utils.DataConverter;
import io.grpc.utils.FileUtils;

import static org.opencv.core.CvType.CV_32F;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private Button trainButton;
    private Button faceUpload;
    private Button faceRec;
    private Button trainFaceUpload;
    private ImageView faceImg;
    private Bitmap bitmap;
    private Context context;
    private ArrayList<String> fileList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        faceUpload = findViewById(R.id.faceUpload);
        faceImg = findViewById(R.id.faceImg);
        faceRec = findViewById(R.id.faceRec);
        trainButton = (Button) findViewById(R.id.train);
        trainFaceUpload = (Button) findViewById(R.id.trainFaceUpload);
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
        faceRec.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Bitmap bit = bitmap.copy(Bitmap.Config.ARGB_8888, false);
                Mat src = new Mat(bit.getHeight(), bit.getWidth(), CvType.CV_8UC(3));
                Utils.bitmapToMat(bit, src);
                Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
                for (String filePath : fileList) {
                    Mat image = Imgcodecs.imread(cacheFile(filePath).getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
                    float[][][][] floats = DataConverter.cvMat_3dArray(image, 1);
                }
                Utils.matToBitmap(src, bitmap);
                Message message = new Message();
                message.what = 1;
                handler.sendMessage(message);
            }
        });

    }

    public File cacheFile(String filename) {
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

    public static boolean saveBitmapToSd(Bitmap bitmap, String filePath) {
        FileOutputStream outputStream;
        outputStream = null;
        try {
            File file = new File(filePath);
            if (file.exists() || file.isDirectory()) {
                file.delete();
            }
            file.createNewFile();
            outputStream = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 0, outputStream);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        } finally {
            if (outputStream != null) {
                try {
                    outputStream.flush();
                    outputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return true;
    }

    public void Training(View view) {
        trainButton.setEnabled(false);
        new Training.LocalTraining(this, this.context).execute(
                "123"
        );
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
