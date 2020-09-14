package io.grpc.learning.api;

import org.tensorflow.Graph;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;


public class LogisticsRegression extends BaseGraph {

    public String pbPath;
    /**
     * @param loadPBFile load pb from disk or not
     * @param pyDir      to refer to the python script to create the graph.
     * @param pbDisk
     */
    public LogisticsRegression(boolean loadPBFile, String pyDir, Boolean pbDisk) {
        if (loadPBFile) {
            if (pbDisk){
                this.pbPath = this.pbDiskPath+ '/' +LogisticsRegression.class.getSimpleName() + this.pbSuffix;
            }
            else{
                StringBuffer sBuffer = new StringBuffer(this.toString().replace(".", "_"));
                sBuffer.append(pbSuffix);
                this.buildGraph(pyDir, sBuffer.toString());
                this.pbPath = String.format("%s/%s", pbFile.getAbsolutePath(), sBuffer.toString());
            }
            this.loadGraph();
        } else {
            // todo: create graph using java code
            Graph graph = new Graph();
            this.graph = graph;
        }
    }

    public LogisticsRegression(){
        this(true,"src/main/python/LogisticsRegression.py", true);
    }
    public LogisticsRegression(Boolean pdDisk){
        this(true,"src/main/python/LogisticsRegression.py", pdDisk);
    }

    private void loadGraph() {
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            modelStream = new FileInputStream(this.pbPath);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = modelStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }

            buffer.flush();
            byte[] byteArray = buffer.toByteArray();
            graph.importGraphDef(byteArray);
            this.graph = graph;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void deletePBFile()  {
        pbFile.delete();
        File dir = null;
        try {
            dir = new File(pbFile.getCanonicalPath());
            File[] listFiles = dir.listFiles();
            System.out.println("Cleaning out folder:" + dir.toString());
            for (File file : listFiles){
                file.delete();
            }
            dir.delete();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
