package io.grpc.api;

import android.widget.TextView;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import io.grpc.utils.LocalCSVReader;
import io.grpc.vo.FeedDict;
import io.grpc.vo.SequenceType;
import io.grpc.vo.TrainableVariable;

public class SessionRunner {
    private Graph graph;
    private int epoch;
    private int batchSize;
    private Session session;
    private TrainInitialize trainInitialize;
    private LocalCSVReader localCSVReader;
    private TrainableVariable trainableVariable;
    private Tensor optimizer;
    private String optimizerName = "loss";
    private FeedDict feedDict = new FeedDict();

    public SessionRunner(Graph graph, LocalCSVReader localCSVReader, int epoch, int batchSize) {
        this.graph = graph;
        this.epoch = epoch;
        this.batchSize = batchSize;
        this.localCSVReader = localCSVReader;
        this.session = new Session(this.graph);
    }

    public SessionRunner(Graph graph, SequenceType sequenceType,
                         LocalCSVReader localCSVReader, int epoch) {
        this.graph = graph;
        this.epoch = epoch;
        this.localCSVReader = localCSVReader;
        this.trainInitialize = new TrainInitialize(this.localCSVReader);
        this.VariablesProducer(sequenceType.getTensorVar(), sequenceType.getTensorName(),
                sequenceType.getTensorTargetName(), sequenceType.getTensorShape(),
                sequenceType.getTensorAssignName());
        this.feedTrainData(sequenceType.getPlaceholder());
        this.session = new Session(this.graph);
    }

    /**
     *
     */
    public void invoke(TextView textView) {
        this.globalVariablesInitializer();
        this.train(textView);
    }

    private void train(TextView textView) {
        int epoch = this.epoch;
        int batchSize = this.batchSize;
        Session.Runner runner = this.session.runner().fetch(this.optimizerName);
        for (String s : feedDict.getStringList()) {
            if (feedDict.getFeed2DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDict.getFeed2DData().get(s)));
            } else if (feedDict.getFeed1DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDict.getFeed1DData().get(s)));
            } else {
                runner = runner.feed(s, Tensor.create(feedDict.getFeedFloat().get(s)));
            }
        }
        this.optimizer = runner.run().get(0);
        this.updateVariables();
        textView.setText(String.valueOf("Loss is: " + this.optimizer.floatValue()));

    }

    private void feedTrainData(List<String> stringList) {
        feedDict.setStringList(stringList);
        feedDict.getFeedFloat().put(stringList.get(0), (float) this.trainInitialize.getX().length);
        feedDict.getFeed2DData().put(stringList.get(1), this.trainInitialize.getX());
        feedDict.getFeed2DData().put(stringList.get(2), this.trainInitialize.getY_oneHot());
    }

    private void VariablesProducer(List<List<Float>> trainableVar, List<String> trainableVarName,
                                   List<String> targetVarName, List<List<Integer>> listShape,
                                   List<String> assignVarName) {
        assert this.trainInitialize != null;
        trainableVariable = this.trainInitialize.initVar(trainableVar, trainableVarName,
                targetVarName, assignVarName, listShape);
    }

    /**
     *
     */
    private void globalVariablesInitializer() {
        HashMap<String, String> nameMap = trainableVariable.getNameMap();
        HashMap<String, float[][]> weight = trainableVariable.getWeight();
        HashMap<String, float[]> bias = trainableVariable.getBias();
        Set<String> keySet = new HashSet<String>();
        keySet.addAll(weight.keySet());
        keySet.addAll(bias.keySet());
        for (String s : keySet) {
            if (weight.keySet().contains(s)) {
                this.session.runner().feed(s, Tensor.create(weight.get(s)))
                        .addTarget(nameMap.get(s))
                        .run();
            } else {
                this.session.runner().feed(s, Tensor.create(bias.get(s)))
                        .addTarget(nameMap.get(s))
                        .run();
            }
        }
    }

    /**
     * When run this function, remember epoch -1
     */
    private void updateVariables() {
        List<String> backPropagationList = trainableVariable.getBackPropagationList();
        FeedDict feedDict = this.feedDict;
        Session.Runner runner = this.session.runner();
        for (String s : feedDict.getStringList()) {
            if (feedDict.getFeed2DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDict.getFeed2DData().get(s)));
            } else if (feedDict.getFeed1DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDict.getFeed1DData().get(s)));
            } else {
                runner = runner.feed(s, Tensor.create(feedDict.getFeedFloat().get(s)));
            }
        }
        HashMap<String, String> nameMap = trainableVariable.getNameMap();
        HashMap<String, float[][]> weight = trainableVariable.getWeight();
        HashMap<String, float[]> bias = trainableVariable.getBias();
        for (String assignVar : backPropagationList) {
            Tensor tensor = runner.addTarget(assignVar).run().get(0);
        }
    }
}
