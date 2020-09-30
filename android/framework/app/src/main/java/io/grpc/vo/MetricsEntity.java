package io.grpc.vo;

public class MetricsEntity {

    public String preName = "pre/Variable";
    public String lossName = "loss";
    public String AUCName = "";

    private float loss;
    private float AUC;

    private float eval_loss;

    public float getEval_loss() {
        return eval_loss;
    }

    public void setEval_loss(float eval_loss) {
        this.eval_loss = eval_loss;
    }

    public float getEval_AUC() {
        return eval_AUC;
    }

    public void setEval_AUC(float eval_AUC) {
        this.eval_AUC = eval_AUC;
    }

    private float eval_AUC;

    public float getLoss() {
        return loss;
    }

    public void setLoss(float loss) {
        this.loss = loss;
    }

    public float getAUC() {
        return AUC;
    }

    public void setAUC(float AUC) {
        this.AUC = AUC;
    }
}
