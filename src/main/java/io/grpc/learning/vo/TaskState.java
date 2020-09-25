package io.grpc.learning.vo;

public enum TaskState {
    /**
     *
     */
    call("call", true), 
    callValue("callValue", true), 
    sendValue("sendValue", true);
    private String name;
    private boolean state;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isState() {
        return state;
    }

    public void setState(boolean state) {
        this.state = state;
    }

    TaskState(String name, boolean state) {
        this.name = name;
        this.state = state;
    }
}
