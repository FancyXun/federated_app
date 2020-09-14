package io.grpc.vo;

import java.util.HashMap;
import java.util.List;

public class FeedDict {
    private HashMap<String, float[][]> feed2DData;
    private HashMap<String, float[]> feed1DData;
    private HashMap<String, Integer> feedInt;

    public List<String> getStringList() {
        return stringList;
    }

    public void setStringList(List<String> stringList) {
        this.stringList = stringList;
    }

    private List<String> stringList;

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

    public HashMap<String, Integer> getFeedInt() {
        return feedInt;
    }

    public void setFeedInt(HashMap<String, Integer> feedInt) {
        this.feedInt = feedInt;
    }
}
