package io.grpc.vo;

public class ImageInfo {
    int batch_size = 16;
    float total_loss = 0;
    int height = 112;
    int width = 112;
    int channel = 3;
    int label_num = 1006;

    public int getBatch_size() {
        return batch_size;
    }

    public float getTotal_loss() {
        return total_loss;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public int getChannel() {
        return channel;
    }

    public int getLabel_num() {
        return label_num;
    }
}
