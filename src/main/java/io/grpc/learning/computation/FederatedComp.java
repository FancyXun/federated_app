package io.grpc.learning.computation;

import org.tensorflow.Graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.grpc.learning.api.BaseGraph;
import io.grpc.learning.vo.GraphZoo;
import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.TaskZoo;

public class FederatedComp {

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

    /**
     *
     * @param nodeName
     * @return
     */
    public static SequenceData aggregation(String nodeName){
        HashMap<String, HashMap<String, SequenceData>> weightAll = TaskZoo.getTaskQueue();
        List<List<Float>> weightFloat ;
        List<List<Float>> weightFloat1 ;
        Iterator<Map.Entry<String, HashMap<String, SequenceData>>> iterator = weightAll.entrySet().iterator();
        Map.Entry<String, HashMap<String, SequenceData>> firstEle = iterator.next();
        HashMap<String, SequenceData> weightAgg = firstEle.getValue();
        weightFloat = weightAgg.get(nodeName).getTensorVar();;
        for (; iterator.hasNext();){
            weightFloat1 = iterator.next().getValue().get(nodeName).getTensorVar();
            for (int i =0 ; i< weightFloat1.size(); i++){
                List<Float> List = weightFloat.get(i);
                List<Float> List1 = weightFloat1.get(i);
                List<Float> finalList = IntStream.range(0, List.size())
                        .mapToObj(j -> List.get(j) + List1.get(j))
                        .collect(Collectors.toList());
                weightFloat.set(i, finalList);
            }
        }
        return weightAgg.get(nodeName);
    }


}
