package io.grpc.learning.vo;

import java.util.HashMap;
import java.util.List;

public class RoundStateInfo {
    public static HashMap<String, List<String>> callRequest = new HashMap<>();
    public static HashMap<String, List<String>> callValueRequest = new HashMap<>();
    public static HashMap<String, List<String>> collectValueRequest = new HashMap<>();

    public static HashMap<String, StateMachine> RoundState = new HashMap<String, StateMachine>();


}
