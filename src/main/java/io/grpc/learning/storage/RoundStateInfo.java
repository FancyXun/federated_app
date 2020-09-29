package io.grpc.learning.storage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import io.grpc.learning.computation.TensorValue;
import io.grpc.learning.vo.SequenceData;
import io.grpc.learning.vo.StateMachine;

public class RoundStateInfo {

    public static int round = 100;
    public static int maxRound = 100;
    public static String dataSplit = "train@0-8";
    public static HashMap<String, List<String>> callRequest = new HashMap<>();
    public static HashMap<String, Set<String>> waitRequest = new HashMap<>();
    public static HashMap<String, StateMachine> roundState = new HashMap<>();
    public static HashMap<String, Integer> epochMap = new HashMap<>();

    /**
     * @param node
     * @param clientId
     */
    public static void callUpdate(String node, String clientId) {
        if (callRequest.get(node) == null) {
            callRequest.put(node, new ArrayList<>());
            waitRequest.put(node, new HashSet<>());
            MetricsContainer.MetricsCollector.put(node, new HashMap<>());
        }
        callRequest.get(node).add(clientId);
        epochMap.put(clientId, round);
        if (!roundState.containsKey(node)) {
            roundState.put(node, StateMachine.start);
        }
    }

    /**
     * @param req
     */
    public static void collectValueUpdate(TensorValue req) {
        String node = req.getNodeName();
        String clientId = req.getId();
        // First request , change state machine to wait
        if (waitRequest.get(node).isEmpty()) {
            roundState.put(node, StateMachine.wait);
        }
//        else {
//            for (String key : epochMap.keySet()) {
//                if (epochMap.get(key) != round) {
//                    waitRequest.get(node).remove(key);
//                }
//            }
//        }
        SequenceData sequenceData = ModelWeights.weightsCollector.get(clientId).get(node);
        sequenceData.getTensorVar().set(req.getOffset(), req.getListArrayList());
        ModelWeights.weightsCollector.get(req.getId()).put(node, sequenceData);
        if (req.getOffset() == req.getValueSize() - 1) {
            waitRequest.get(node).add(clientId);
            MetricsContainer.MetricsCollector.get(node).put(clientId,req.getMetrics());
        }
    }

    public static void dropoutClients() {

    }

}
