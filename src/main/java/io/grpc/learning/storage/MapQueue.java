package io.grpc.learning.storage;

import java.util.Set;

import io.grpc.learning.computation.FederatedComp;
import io.grpc.learning.vo.StateMachine;

public class MapQueue {
    /**
     *
     * check waiting queue if a request call in this round, If state is wait,
     * the request should wait until it changes to others.
     * @param name
     * @param val
     */
    public static void queueChecker(String name, String val){
        Set<String> waitQueue = RoundStateInfo.waitRequest.get(name);
        if (waitQueue != null
                && waitQueue.contains(val)) {
            while (RoundStateInfo.roundState.get(name) == StateMachine.wait) {
                System.out.println(val + " " + StateMachine.wait + "...");
                FederatedComp.timeWait(1000);
            }
            waitQueue.remove(val);
            // wait client delete self from queue
//                while (!RoundStateInfo.waitRequest.get(nodeName).isEmpty()){
//                    FederatedComp.timeWait(1000);
//                    System.out.println(RoundStateInfo.roundState.get(nodeName));
//                    System.out.println(RoundStateInfo.waitRequest.get(nodeName));
//                }
        }
    }
}
