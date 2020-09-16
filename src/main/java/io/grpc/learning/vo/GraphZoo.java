package io.grpc.learning.vo;

import org.tensorflow.Graph;

import java.util.HashMap;

public class GraphZoo {
    private volatile static GraphZoo instance = null;

    public static GraphZoo getInstance() {
        if (instance == null) {
            synchronized (GraphZoo.class) {
                if (instance == null) {
                    instance = new GraphZoo();
                }
            }

        }
        return instance;
    }


    public HashMap<String, Graph> getGraphZoo() {
        return graphZoo;
    }

    public void setGraphZoo(HashMap<String, Graph> graphZoo) {
        GraphZoo.graphZoo = graphZoo;
    }

    private static HashMap<String, Graph> graphZoo = new HashMap<>();
}
