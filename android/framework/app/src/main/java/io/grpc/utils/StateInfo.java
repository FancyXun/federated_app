package io.grpc.utils;

/**
 *
 */
public class StateInfo {

    private static StateInfo stateInfo = null;

    public StateInfo() {
    }

    public static StateInfo getInstance() {
        if (stateInfo == null) {
            stateInfo = new StateInfo();
        }
        return stateInfo;
    }

    public int getStateCode() {
        return stateCode;
    }

    public void setStateCode(int stateCode) {
        this.stateCode = stateCode;
    }

    private int stateCode = 0;
}
