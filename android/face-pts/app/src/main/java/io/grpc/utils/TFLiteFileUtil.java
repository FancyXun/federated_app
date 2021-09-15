package io.grpc.utils;
import android.content.Context;
import android.content.res.AssetFileDescriptor;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.net.URL;
import java.net.URLConnection;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;


public class TFLiteFileUtil {
    private TFLiteFileUtil() {}

    /**
     * Loads labels from the label file into a list of strings.
     *
     * <p>A legal label file is the plain text file whose contents are split into lines, and each line
     * is an individual value. The file should be in assets of the context.
     *
     * @param context The context holds assets.
     * @param filePath The path of the label file, relative with assets directory.
     * @return a list of labels.
     * @throws IOException if error occurs to open or read the file.
     */
    public static List<String> loadLabels( Context context, String filePath)
            throws IOException {
        return loadLabels(context, filePath, Charset.defaultCharset());
    }

    /**
     * Loads labels from the label file into a list of strings.
     *
     * <p>A legal label file is the plain text file whose contents are split into lines, and each line
     * is an individual value. The empty lines will be ignored. The file should be in assets of the
     * context.
     *
     * @param context The context holds assets.
     * @param filePath The path of the label file, relative with assets directory.
     * @param cs {@code Charset} to use when decoding content of label file.
     * @return a list of labels.
     * @throws IOException if error occurs to open or read the file.
     */
    
    public static List<String> loadLabels(
             Context context,  String filePath, Charset cs) throws IOException {
        try (InputStream inputStream = context.getAssets().open(filePath)) {
            return loadLabels(inputStream, cs);
        }
    }

    /**
     * Loads labels from an input stream of an opened label file. See details for label files in
     *
     * @param inputStream the input stream of an opened label file.
     * @return a list of labels.
     * @throws IOException if error occurs to open or read the file.
     */
    
    public static List<String> loadLabels( InputStream inputStream) throws IOException {
        return loadLabels(inputStream, Charset.defaultCharset());
    }

    /**
     * Loads labels from an input stream of an opened label file. See details for label files in
     *
     * @param inputStream the input stream of an opened label file.
     * @param cs {@code Charset} to use when decoding content of label file.
     * @return a list of labels.
     * @throws IOException if error occurs to open or read the file.
     */
    
    public static List<String> loadLabels( InputStream inputStream, Charset cs)
            throws IOException {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, cs))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().length() > 0) {
                    labels.add(line);
                }
            }
            return labels;
        }
    }

    /**
     * Loads a vocabulary file (a single-column text file) into a list of strings.
     *
     * <p>A vocabulary file is a single-column plain text file whose contents are split into lines,
     * and each line is an individual value. The file should be in assets of the context.
     *
     * @param context The context holds assets.
     * @param filePath The path of the vocabulary file, relative with assets directory.
     * @return a list of vocabulary words.
     * @throws IOException if error occurs to open or read the file.
     */
    
    public static List<String> loadSingleColumnTextFile(
             Context context,  String filePath, Charset cs) throws IOException {
        return loadLabels(context, filePath, cs);
    }

    /**
     * Loads vocabulary from an input stream of an opened vocabulary file (which is a single-column
     * String)}.
     *
     * @param inputStream the input stream of an opened vocabulary file.
     * @return a list of vocabulary words.
     * @throws IOException if error occurs to open or read the file.
     */
    
    public static List<String> loadSingleColumnTextFile( InputStream inputStream, Charset cs)
            throws IOException {
        return loadLabels(inputStream, cs);
    }

    /**
     * Loads a file from the asset folder through memory mapping.
     *
     * @param context Application context to access assets.
     * @param filePath Asset path of the file.
     * @return the loaded memory mapped file.
     * @throws IOException if an I/O error occurs when loading the tflite model.
     */
    
    public static MappedByteBuffer loadMappedFile( Context context,  String filePath)
            throws IOException {


        File file = new File(filePath);
        long len = file.length();
        MappedByteBuffer mappedByteBuffer = null;
        try {
            mappedByteBuffer = new RandomAccessFile(file, "r")
                    .getChannel()
                    .map(FileChannel.MapMode.READ_ONLY, 0, len);

        } catch (IOException e) {
            e.printStackTrace();
        }

        assert mappedByteBuffer != null;
        return mappedByteBuffer;
    }


    public static MappedByteBuffer loadMapperFileFromContext(Context context, String filepath)
        throws IOException{
                try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(filepath);
                     FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    /**
     * Loads a binary file from the asset folder.
     *
     * @param context Application context to access assets.
     * @param filePath Asset path of the file.
     * @return the byte array for the binary file.
     * @throws IOException if an I/O error occurs when loading file.
     */
    
    public static byte[] loadByteFromFile( Context context,  String filePath)
            throws IOException {
        ByteBuffer buffer = loadMappedFile(context, filePath);
        byte[] byteArray = new byte[buffer.remaining()];
        buffer.get(byteArray);
        return byteArray;
    }
    /**
     * Download tf lite from server and save to tmp file
     *
     * @param url tf lite model url in server
     * @param outputFile tmp path of the file.
     */
    public static void downloadFile(String url, File outputFile) {
        try {
            URL u = new URL(url);
            URLConnection conn = u.openConnection();
            int contentLength = conn.getContentLength();
            DataInputStream stream = new DataInputStream(u.openStream());
            byte[] buffer = new byte[contentLength];
            stream.readFully(buffer);
            stream.close();

            DataOutputStream fos = new DataOutputStream(new FileOutputStream(outputFile));
            fos.write(buffer);
            fos.flush();
            fos.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }
}
