package io.grpc.learning.api;

import org.tensorflow.Graph;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import sun.misc.IOUtils;

public class LogisticsRegression extends BaseGraph {

    public String pbPath;
    /**
     * @param loadPBFile load pb from disk or not
     * @param pyDir      to refer to the python script to create the graph.
     */
    public LogisticsRegression(boolean loadPBFile, String pyDir) {
        if (loadPBFile) {
            StringBuffer sBuffer = new StringBuffer(this.toString().replace(".", "_"));
            sBuffer.append(pbSuffix);
            this.buildGraph(pyDir, sBuffer.toString());
            this.pbPath = String.format("%s/%s", pbFile.getAbsolutePath(), sBuffer.toString());
            this.loadGraph();
        } else {
            // todo: create graph using java code
            Graph graph = new Graph();
            this.graph = graph;
        }
    }

    private void loadGraph() {
        Graph graph = new Graph();
        InputStream modelStream = null;
        try {
            modelStream = new FileInputStream(this.pbPath);
            graph.importGraphDef(IOUtils.readAllBytes(modelStream));
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
