package io.grpc.learning.utils;

import org.junit.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FileUtils {

    @Test
    public void appendFile(){
        File f = new File("/tmp/model_weights/1.txt");
        if (!f.exists()) {
            try {
                f.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(f, true));
            bw.write("request.getTensor().getFloatValList())");
            bw.close();
        }catch(IOException e){
            e.printStackTrace();
        }
    }

}
