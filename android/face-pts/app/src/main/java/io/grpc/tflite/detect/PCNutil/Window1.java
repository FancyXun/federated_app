package io.grpc.tflite.detect.PCNutil;

public class Window1 {
    private int x;
    private int y;
    private int width;
    private double angle;
    private double score;

    public Window1() {
    }

    public Window1(int x, int y, int width, double angle, double score) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.angle = angle;
        this.score = score;
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

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public double getAngle() {
        return angle;
    }

    public void setAngle(double angle) {
        this.angle = angle;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public String toString() {
        return "Window1{" +
                "x=" + x +
                ", y=" + y +
                ", width=" + width +
                ", angle=" + angle +
                ", score=" + score +
                '}';
    }
}
