package io.grpc.utils;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;

// get local file list, example:
// ArrayList<String> fileList = new FileUtils(context, "sampleData/casiaWebFace").getFileList();
public class FileUtils {

    private ArrayList<String> fileList = new ArrayList<>();
    private Context context;

    public ArrayList<String> getFileList() {
        return fileList;
    }

    public FileUtils(Context context, String dirPath) {
        this.context = context;
        listOfFiles(dirPath);
    }

    private void listOfFiles(String dirPath) {
        try {
            String[] files = context.getAssets().list(dirPath);
            assert files != null;
            if (files.length == 0) {
                fileList.add(dirPath);
            } else {
                for (String f : files) {
                    listOfFiles(dirPath + "/" + f);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
