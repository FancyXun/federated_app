package io.grpc.vo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class FeedDict {
    private HashMap<String, float[][]> feed2DData = new HashMap<>();
    private HashMap<String, float[]> feed1DData = new HashMap<>();

    public HashMap<String, Float> getFeedFloat() {
        return feedFloat;
    }

    public void setFeedFloat(HashMap<String, Float> feedFloat) {
        this.feedFloat = feedFloat;
    }

    private HashMap<String, Float> feedFloat = new HashMap<>();

    public List<String> getStringList() {
        return stringList;
    }

    public void setStringList(List<String> stringList) {
        this.stringList = stringList;
    }

    private List<String> stringList = new ArrayList<>();

    public HashMap<String, float[][]> getFeed2DData() {
        return feed2DData;
    }

    public void setFeed2DData(HashMap<String, float[][]> feed2DData) {
        this.feed2DData = feed2DData;
    }

    public HashMap<String, float[]> getFeed1DData() {
        return feed1DData;
    }

    public void setFeed1DData(HashMap<String, float[]> feed1DData) {
        this.feed1DData = feed1DData;
    }


}
