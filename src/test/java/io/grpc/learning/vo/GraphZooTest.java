package io.grpc.learning.vo;

import org.junit.Test;
import org.tensorflow.Graph;

public class GraphZooTest {
    @Test
    public void GraphZooTest(){
        GraphZoo graphZoo = new GraphZoo();
        Graph graph = graphZoo.getGraphZoo().get("graph1");
        System.out.println(graphZoo.getGraphZoo());
        System.out.println(graphZoo.getGraphZoo().get("graph1"));
        System.out.println(graph == null);
        graphZoo.getGraphZoo().put("graph1",new Graph());
        System.out.println(graphZoo.getGraphZoo());
        GraphZoo graphZoo1 = new GraphZoo();
        graphZoo.getGraphZoo().put("graph2",new Graph());
        System.out.println(graphZoo1.getGraphZoo());
        GraphZoo graphZoo2 = new GraphZoo();
        System.out.println(graphZoo1.getGraphZoo());

    }
}
