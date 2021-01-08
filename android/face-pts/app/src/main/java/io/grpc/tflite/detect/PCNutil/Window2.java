package io.grpc.tflite.detect.PCNutil;

public class Window2 {
    private int x;
    private int y;
    private int w;
    private int h;
    private double angle;
    private double scale;
    private float conf;

    public Window2() {
    }

    public Window2(int x, int y, int w, int h, double angle, double scale, float conf) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.angle = angle;
        this.scale = scale;
        this.conf = conf;
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public int getW() {
        return w;
    }

    public void setW(int w) {
        this.w = w;
    }

    public int getH() {
        return h;
    }

    public void setH(int h) {
        this.h = h;
    }

    public double getAngle() {
        return angle;
    }

    public void setAngle(double angle) {
        this.angle = angle;
    }

    public double getScale() {
        return scale;
    }

    public void setScale(double scale) {
        this.scale = scale;
    }

    public float getConf() {
        return conf;
    }

    public void setConf(float conf) {
        this.conf = conf;
    }

    @Override
    public String toString() {
        return "Window2{" +
                "x=" + x +
                ", y=" + y +
                ", w=" + w +
                ", h=" + h +
                ", angle=" + angle +
                ", scale=" + scale +
                ", conf=" + conf +
                '}';
    }
}
