package io.grpc.learning.vo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import io.grpc.learning.computation.TensorValue;

public class RoundStateInfo {

    public static int round = 10;


    public static HashMap<String, List<String>> callRequest = new HashMap<>();
    public static HashMap<String, Set<String>> waitRequest = new HashMap<>();
    public static HashMap<String, List<String>> callValueRequest = new HashMap<>();
    public static HashMap<String, List<String>> collectValueRequest = new HashMap<>();

    public static HashMap<String, StateMachine> roundState = new HashMap<>();

    public static HashMap<String, Integer> epochMap = new HashMap<>();

    /**
     * @param node
     * @param clientId
     */
    public static void callUpdate(String node, String clientId) {
        if (RoundStateInfo.callRequest.get(node) == null) {
            RoundStateInfo.callRequest.put(node, new ArrayList<>());
            RoundStateInfo.waitRequest.put(node, new HashSet<>());
        }
        RoundStateInfo.callRequest.get(node).add(clientId);
        RoundStateInfo.epochMap.put(node, round);
        if (!RoundStateInfo.roundState.containsKey(node)) {
            RoundStateInfo.roundState.put(node, StateMachine.start);
        }
    }

    /**
     * @param req
     */
    public static void collectValueUpdate(TensorValue req) {
        String node = req.getNodeName();
        String clientId = req.getId();
        // First request , change state machine to wait
        if (RoundStateInfo.collectValueRequest.isEmpty() ) {
            while (!RoundStateInfo.waitRequest.get(node).isEmpty()){
                try {
                    if (RoundStateInfo.roundState.get(node) == StateMachine.wait){
                        RoundStateInfo.roundState.put(node, StateMachine.start);
                    }
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            RoundStateInfo.roundState.put(node, StateMachine.wait);
            RoundStateInfo.collectValueRequest.put(node, new ArrayList<>());
        }
        SequenceData sequenceData = ModelWeights.weightsCollector.get(clientId).get(node);
        sequenceData.getTensorVar().set(req.getOffset(), req.getListArrayList());
        ModelWeights.weightsCollector.get(req.getId()).put(node, sequenceData);
        if (req.getOffset() == req.getValueSize() - 1) {
            RoundStateInfo.collectValueRequest.get(node).add(clientId);
        }
    }

    public static void dropoutClients(){

    }

}
