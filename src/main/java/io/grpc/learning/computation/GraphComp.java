package io.grpc.learning.computation;

import org.tensorflow.Graph;

import io.grpc.learning.api.BaseGraph;
import io.grpc.learning.vo.GraphZoo;

public class GraphComp {

    private static final String url = "io.grpc.learning.api";

    /**
     * @param node_name
     * @return
     */
    public static Graph getGraph(String node_name) {
        GraphZoo graphZoo = new GraphZoo();
        Graph graph = graphZoo.getGraphZoo().get(node_name);
        if (graph == null) {
            try {
                ClassLoader classLoader = Class.forName(url + "." + node_name).getClassLoader();
                BaseGraph basegraph = (BaseGraph) classLoader.loadClass(url + "." + node_name).newInstance();
                graph = basegraph.getGraph();
                graphZoo.getGraphZoo().put(node_name, graph);
                GraphZoo.getGraphZooPath().put(node_name, basegraph.pbPath);
            } catch (Exception ClassNotFoundException) {
                throw new RuntimeException();
            }
        }
        return graph;
    }
}
