package io.grpc.learning.vo;

import java.util.HashSet;

public class Client {

    private HashSet<String> callTrainingClients = new HashSet<>();
    private HashSet<String> callModelClients = new HashSet<>();
    private HashSet<String> callLayerWeightsClients = new HashSet<>();
    private HashSet<String> computeLayerWeightsClients = new HashSet<>();
    private HashSet<String> computeFinishClients = new HashSet<>();

    public HashSet<String> getCallTrainingClients() {
        return callTrainingClients;
    }

    public void setCallTrainingClients(HashSet<String> callTrainingClients) {
        this.callTrainingClients = callTrainingClients;
    }

    public HashSet<String> getCallModelClients() {
        return callModelClients;
    }

    public void setCallModelClients(HashSet<String> callModelClients) {
        this.callModelClients = callModelClients;
    }

    public HashSet<String> getCallLayerWeightsClients() {
        return callLayerWeightsClients;
    }

    public void setCallLayerWeightsClients(HashSet<String> callLayerWeightsClients) {
        this.callLayerWeightsClients = callLayerWeightsClients;
    }

    public HashSet<String> getComputeLayerWeightsClients() {
        return computeLayerWeightsClients;
    }

    public void setComputeLayerWeightsClients(HashSet<String> computeLayerWeightsClients) {
        this.computeLayerWeightsClients = computeLayerWeightsClients;
    }

    public HashSet<String> getComputeFinishClients() {
        return computeFinishClients;
    }

    public void setComputeFinishClients(HashSet<String> computeFinishClients) {
        this.computeFinishClients = computeFinishClients;
    }
}
