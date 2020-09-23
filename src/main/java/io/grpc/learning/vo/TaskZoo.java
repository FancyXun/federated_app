package io.grpc.learning.vo;

import java.util.HashMap;

public class TaskZoo {
    private volatile static TaskZoo instance = null;
    private static HashMap<String, Integer> TaskInt = new HashMap<>();

    public static TaskZoo getInstance() {
        if (instance == null) {
            synchronized (TaskZoo.class) {
                if (instance == null) {
                    instance = new TaskZoo();

                }
            }

        }
        return instance;
    }

    public static HashMap<String, TaskState> getTask() {
        return Task;
    }

    public static void setTask(HashMap<String, TaskState> task) {
        Task = task;
    }

    private static HashMap<String, TaskState> Task = new HashMap<>();

    public static HashMap<String, HashMap<String, SequenceData>> getTaskQueue() {
        return TaskQueue;
    }

    public static void setTaskQueue(HashMap<String, HashMap<String, SequenceData>> taskQueue) {
        TaskQueue = taskQueue;
    }

    private static HashMap<String, HashMap<String, SequenceData>> TaskQueue = new HashMap<>();

    public static HashMap<String, Integer> getTaskInt() {
        return TaskInt;
    }


}
