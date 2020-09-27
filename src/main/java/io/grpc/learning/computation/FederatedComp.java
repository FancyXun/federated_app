package io.grpc.learning.computation;

import com.alibaba.fastjson.JSON;

import org.tensorflow.Graph;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.grpc.learning.api.BaseGraph;
import io.grpc.learning.utils.JsonUtils;
import io.grpc.learning.vo.GraphZoo;
import io.grpc.learning.vo.ModelWeights;
import io.grpc.learning.vo.RoundStateInfo;
import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.StateMachine;
import io.grpc.learning.vo.TaskZoo;
import io.grpc.learning.vo.TensorVarName;

import static io.grpc.learning.computation.SequenceComp.initializerSequence;

public class FederatedComp implements Runnable {

    private static final String url = "io.grpc.learning.api";
    private static HashMap<String, SequenceData> weight;
    public static boolean update = false;

    /**
     * @param nodeName
     * @return
     */
    public synchronized static Graph getGraph(String nodeName) {
        GraphZoo graphZoo = new GraphZoo();
        Graph graph = graphZoo.getGraphZoo().get(nodeName);
        if (graph == null) {
            try {
                ClassLoader classLoader = Class.forName(url + "." + nodeName).getClassLoader();
                BaseGraph basegraph = (BaseGraph) classLoader.loadClass(url + "." + nodeName).newInstance();
                graph = basegraph.getGraph();
                graphZoo.getGraphZoo().put(nodeName, graph);
                GraphZoo.getGraphZooPath().put(nodeName, basegraph.pbPath);
                GraphZoo.getGraphJsonZooPath().put(nodeName, basegraph.pbJson);
            } catch (Exception ClassNotFoundException) {
                throw new RuntimeException();
            }
        }
        return graph;
    }

//    public synchronized static SequenceData aggregation(String nodeName){
//        if (TaskZoo.getUpdate().get(nodeName)){
//            return weight.get(nodeName);
//        }
//        else{
//            return aggregationInner(nodeName);
//        }
//    }

    /**
     * @param request
     * @return
     */

    public synchronized static boolean aggregationInner(TensorValue request) {
        if (update) {
            return true;
        }
        if (RoundStateInfo.roundState.get(request.getNodeName()) == StateMachine.start) {
            return update;
        }
        String nodeName = request.getNodeName();
        HashMap<String, HashMap<String, SequenceData>> weightAll = ModelWeights.weightsCollector;
        int numWeights = weightAll.size();
        List<List<Float>> weightFloat;
        List<List<Float>> weightFloat1;
        Iterator<Map.Entry<String, HashMap<String, SequenceData>>> iterator = weightAll.entrySet().iterator();
        Map.Entry<String, HashMap<String, SequenceData>> firstEle = iterator.next();
        HashMap<String, SequenceData> weightAgg = firstEle.getValue();
        weightFloat = weightAgg.get(nodeName).getTensorVar();
        for (; iterator.hasNext(); ) {
            weightFloat1 = iterator.next().getValue().get(nodeName).getTensorVar();
            for (int i = 0; i < weightFloat1.size(); i++) {
                List<Float> List = weightFloat.get(i);
                List<Float> List1 = weightFloat1.get(i);
                List<Float> finalList = IntStream.range(0, List.size())
                        .mapToObj(j -> List.get(j) + List1.get(j))
                        .collect(Collectors.toList());
                weightFloat.set(i, finalList);
            }
        }
        for (int i = 0; i < weightFloat.size(); i++) {
            for (int j = 0; j < weightFloat.get(i).size(); j++) {
                weightFloat.get(i).set(j, weightFloat.get(i).get(j) / numWeights);
            }
        }
        update = true;
        RoundStateInfo.round -= 1;
        RoundStateInfo.roundState.put(nodeName, StateMachine.start);
        return update;
    }

    /**
     * Initialize weights for one client
     *
     * @param nodeName
     * @param clientId
     * @return
     */
    public static SequenceData weightsInitializer(String nodeName, String clientId) {
        String s = JsonUtils.readJsonFile(GraphZoo.getGraphJsonZooPath().get(nodeName));
        TensorVarName tensorVarName = JsonUtils.jsonToMap(JSON.parseObject(s));
        SequenceData sequenceData = initializerSequence(tensorVarName);
        HashMap<String, SequenceData> sequenceDataHashMap = new HashMap<>();
        sequenceDataHashMap.put(nodeName, sequenceData);
        ModelWeights.weightsCollector.put(clientId, sequenceDataHashMap);
        return sequenceData;
    }

    public synchronized static void updataWeights() {

    }

    @Override
    public void run() {

    }

    public static void timeWait(int second){
        try {
            Thread.sleep(second);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
