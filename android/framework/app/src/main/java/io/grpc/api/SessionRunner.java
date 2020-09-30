package io.grpc.api;

import android.widget.TextView;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import io.grpc.utils.LocalCSVReader;
import io.grpc.utils.listConvert;
import io.grpc.vo.FeedDict;
import io.grpc.vo.SequenceType;
import io.grpc.vo.TrainableVariable;

public class SessionRunner {
    private Graph graph;
    private int round;
    private int batchSize;
    private Session session;
    private TrainInitialize trainInitialize;
    private LocalCSVReader localCSVReader;
    private TrainableVariable trainableVariable;
    public GraphMetrics graphMetrics = new GraphMetrics();
    private FeedDict feedDict = new FeedDict();
    private FeedDict feedDictVal = new FeedDict();
    private List<List<Integer>> tensorAssignShape;

    public SessionRunner(Graph graph, LocalCSVReader localCSVReader, int round, int batchSize) {
        this.graph = graph;
        this.round = round;
        this.batchSize = batchSize;
        this.localCSVReader = localCSVReader;
        this.session = new Session(this.graph);
    }

    /**
     * @param graph
     * @param sequenceType
     * @param localCSVReader
     * @param round
     */
    public SessionRunner(Graph graph, SequenceType sequenceType,
                         LocalCSVReader localCSVReader, int round) {
        this.graph = graph;
        this.round = round;
        this.localCSVReader = localCSVReader;
        this.trainInitialize = new TrainInitialize(this.localCSVReader);
        this.tensorAssignShape = sequenceType.getTensorAssignShape();
        this.VariablesProducer(sequenceType.getTensorVar(), sequenceType.getTensorName(),
                sequenceType.getTensorTargetName(), sequenceType.getTensorShape(),
                sequenceType.getTensorAssignName());
        this.feedTrainData(sequenceType.getPlaceholder());
        this.feedValData(sequenceType.getPlaceholder());
        this.session = new Session(this.graph);
    }

    /**
     *
     */
    public List<List<Float>> invoke(TextView textView) {
        this.globalVariablesInitializer();
        return this.train(textView);
    }

    public void eval(TextView textView) {
        this.evaluateLoss(textView);
    }

    private List<List<Float>> train(TextView textView) {
        int round = this.round;
        int batchSize = this.batchSize;
        Session.Runner runner = this.session.runner().fetch(graphMetrics.lossName);
        for (String s : feedDict.getStringList()) {
            if (feedDict.getFeed2DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDict.getFeed2DData().get(s)));
            } else if (feedDict.getFeed1DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDict.getFeed1DData().get(s)));
            } else {
                runner = runner.feed(s, Tensor.create(feedDict.getFeedFloat().get(s)));
            }
        }
        float loss = runner.run().get(0).floatValue();
        graphMetrics.setLoss(loss);
        textView.setText(String.valueOf("Loss " + this.round + ": " + loss));
        return this.updateVariables();
    }

    private void evaluateLoss(TextView textView) {
        Session.Runner runner = this.session.runner().fetch(graphMetrics.lossName);
        for (String s : feedDictVal.getStringList()) {
            if (feedDictVal.getFeed2DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDictVal.getFeed2DData().get(s)));
            } else if (feedDictVal.getFeed1DData().containsKey(s)) {
                runner = runner.feed(s, Tensor.create(feedDictVal.getFeed1DData().get(s)));
            } else {
                runner = runner.feed(s, Tensor.create(feedDictVal.getFeedFloat().get(s)));
            }
        }
        float eval_loss = runner.run().get(0).floatValue();
        graphMetrics.setEval_loss(eval_loss);
    }

    private void feedTrainData(List<String> stringList) {
        feedDict.setStringList(stringList);
        feedDict.getFeedFloat().put(stringList.get(0), (float) this.trainInitialize.getX_train().length);
        feedDict.getFeed2DData().put(stringList.get(1), this.trainInitialize.getX_train());
        feedDict.getFeed2DData().put(stringList.get(2), this.trainInitialize.getY_oneHot_train());
    }

    private void feedValData(List<String> stringList) {
        feedDictVal.setStringList(stringList);
        feedDictVal.getFeedFloat().put(stringList.get(0), (float) this.trainInitialize.getX_val().length);
        feedDictVal.getFeed2DData().put(stringList.get(1), this.trainInitialize.getX_val());
        feedDictVal.getFeed2DData().put(stringList.get(2), this.trainInitialize.getY_oneHot_val());
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
     * sgd
     */
    private List<List<Float>> updateVariables() {
        List<List<Float>> tensorVar = new ArrayList<>();
        List<String> backPropagationList = trainableVariable.getBackPropagationList();
        FeedDict feedDict = this.feedDict;
        HashMap<String, String> nameMap = trainableVariable.getNameMap();
        HashMap<String, float[][]> weight = trainableVariable.getWeight();
        HashMap<String, float[]> bias = trainableVariable.getBias();
        for (int i = 0; i < backPropagationList.size(); i++) {
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
            Tensor tensor = runner.fetch(backPropagationList.get(i)).run().get(0);
            List<Integer> shapeList = this.tensorAssignShape.get(i);
            if (shapeList.size() == 1) {
                float[] var = new float[shapeList.get(0)];
                tensor.copyTo(var);
                tensorVar.add(listConvert.listConvert(var));

            }
            if (shapeList.size() == 2) {
                float[][] var = new float[shapeList.get(0)][shapeList.get(1)];
                tensor.copyTo(var);
                tensorVar.add(listConvert.listConvert2D(var));
            }
        }
        return tensorVar;
    }

    private void initVarAssign() {

    }
}
