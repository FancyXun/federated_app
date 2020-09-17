package io.grpc.learning.api;

import com.google.common.io.Files;

import org.tensorflow.Graph;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Locale;
import java.util.ResourceBundle;
import java.util.logging.Logger;

import io.grpc.learning.computation.ComputationServer;

public abstract class BaseGraph {
    private static final Logger logger = Logger.getLogger(ComputationServer.class.getName());
    protected Graph graph;
    private ResourceBundle rb = ResourceBundle.getBundle("resource", Locale.getDefault());
    private String pythonExe = (String) rb.getObject("pythonExe");
    protected String pbDiskPath = (String) rb.getObject("pbDiskPath");
    protected String pbSuffix = ".pb";
    public String pbPath;

    protected static File pbFile = Files.createTempDir();

    public Graph getGraph() {
        return this.graph;
    }


    /**
     * @param pyDir     the python directory for tensorflow graph
     * @param graphName the tensorflow graph pb name
     */
    public void buildGraph(String pyDir, String graphName) {
        Process process;
        try {
            logger.info("Server build graph for " + pbFile.getAbsolutePath() + "/" + graphName);
            process = Runtime.getRuntime().exec(String.format("%s %s -p %s -g_name %s", pythonExe, pyDir, pbFile.getAbsolutePath(), graphName));
            BufferedReader in = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            process.destroy();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
