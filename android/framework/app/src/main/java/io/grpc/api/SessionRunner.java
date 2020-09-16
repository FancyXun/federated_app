package io.grpc.api;

import android.widget.TextView;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.HashMap;
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
    private String optimizerName;
    private FeedDict feedDict;

    public SessionRunner(Graph graph, LocalCSVReader localCSVReader, int epoch, int batchSize) {
        this.graph = graph;
        this.epoch = epoch;
        this.batchSize = batchSize;
        this.localCSVReader = localCSVReader;
        this.session = new Session(this.graph);
        this.feedTrainData();
    }

    public SessionRunner(Graph graph, SequenceType sequenceType,
                         LocalCSVReader localCSVReader, int epoch) {
        this.graph = graph;
        this.epoch = epoch;
        this.localCSVReader = localCSVReader;
        this.VariablesProducer(sequenceType.getTensorVar(), sequenceType.getTensorName(),
                sequenceType.getTensorTargetName(), sequenceType.getTensorShape());
        this.session = new Session(this.graph);
        this.feedTrainData();
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
        FeedDict feedDict = this.feedDict;
        for (int i = 0; i < epoch; i++) {
            Session.Runner runner = this.session.runner().fetch(this.optimizerName);
            for (String s : feedDict.getStringList()) {
                if (feedDict.getFeed2DData().containsKey(s)) {
                    runner = runner.feed(s, Tensor.create(feedDict.getFeed2DData().get(s)));
                } else if (feedDict.getFeed1DData().containsKey(s)) {
                    runner = runner.feed(s, Tensor.create(feedDict.getFeed1DData().get(s)));
                } else {
                    runner = runner.feed(s, Tensor.create(feedDict.getFeedInt().get(s)));
                }
            }
            this.optimizer = runner.run().get(0);
            this.updateVariables();
            textView.setText(String.valueOf("the cost of epoch " + i + "/" + epoch + " is: " + this.optimizer.floatValue()));
        }
    }

    private void feedTrainData() {
        this.trainInitialize = new TrainInitialize(this.localCSVReader);
    }

    private void VariablesProducer(List<List<Float>> trainableVar, List<String> trainableVarName,
                                   List<String> targetVarName, List<List<Integer>> listShape) {
        assert this.trainInitialize != null;
        trainableVariable = this.trainInitialize.initVar(trainableVar, trainableVarName, targetVarName, listShape);
    }

    /**
     *
     */
    private void globalVariablesInitializer() {
        HashMap<String, String> nameMap = trainableVariable.getNameMap();
        HashMap<String, float[][]> weight = trainableVariable.getWeight();
        HashMap<String, float[]> bias = trainableVariable.getBias();
        Set<String> weightKeySet = weight.keySet();
        for (String s : nameMap.keySet()) {
            if (weightKeySet.contains(s)) {
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
        Session.Runner runner = null;
        for (String s : feedDict.getStringList()) {
            if (feedDict.getFeed2DData().containsKey(s)) {
                runner = this.session.runner().feed(s, Tensor.create(feedDict.getFeed2DData().get(s)));
            } else if (feedDict.getFeed1DData().containsKey(s)) {
                runner = this.session.runner().feed(s, Tensor.create(feedDict.getFeed1DData().get(s)));
            } else {
                runner = this.session.runner().feed(s, Tensor.create(feedDict.getFeedInt().get(s)));
            }
        }

        for (String s : backPropagationList) {
            assert runner != null;
            runner.addTarget(s).run();
        }
    }
}
