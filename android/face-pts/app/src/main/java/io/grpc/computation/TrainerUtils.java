package io.grpc.computation;

import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import computation.TensorEntity;
import io.grpc.learning.computation.ClientRequest;
import io.grpc.learning.computation.ComputationGrpc;
import io.grpc.learning.computation.Layer;
import io.grpc.learning.computation.ModelWeights;
import io.grpc.learning.computation.ValueReply;

@Deprecated
public class TrainerUtils {
    @Deprecated
    public void getModelWeights(ComputationGrpc.ComputationBlockingStub stub,
                                ClientRequest.Builder builder) {
        ModelWeights modelWeights = stub.callModelWeights(builder.build());
    }

    @Deprecated
    public void setWeights(List<Layer> layerList, ModelWeights modelWeights, Session session) {
        for (int i = 0; i < layerList.size(); i++) {
            TensorEntity.TensorProto tensorProto = modelWeights.getTensor(i);
            Layer layer = layerList.get(i);
            List<Float> floatList = tensorProto.getFloatValList();
            float[] floatArray = new float[floatList.size()];
            int j = 0;
            for (Float f : floatList) {
                floatArray[j++] = (f != null ? f : Float.NaN);
            }
            int dim_count = tensorProto.getTensorShape().getDimCount();
            Tensor tensor = Tensor.create(TrainerStreamUtils.getShape(dim_count,
                    tensorProto.getTensorShape()), FloatBuffer.wrap(floatArray));
            session.runner().feed(layer.getLayerInitName(), tensor)
                    .addTarget(layer.getLayerName() + "/Assign")
                    .run();
        }
    }

    @Deprecated
    public void computeWeights(List<Layer> layerList, int layer_size, Session session,
                               ComputationGrpc.ComputationBlockingStub stub) {
        ModelWeights.Builder modelWeightsBuilder = getWeights(layerList, layer_size, session);
        ValueReply valueReply = stub.computeWeights(modelWeightsBuilder.build());
    }

    @Deprecated
    public ModelWeights.Builder getWeights(List<Layer> layerList, int layer_size, Session session) {
        ModelWeights.Builder modelWeightsBuilder = ModelWeights.newBuilder();
        Pattern p = Pattern.compile("\\d+");
        for (int i = 0; i < layer_size; i++) {
            TensorEntity.TensorProto.Builder tensorBuilder =
                    TensorEntity.TensorProto.newBuilder();
            TensorEntity.TensorShapeProto.Builder tensorShapeBuilder =
                    TensorEntity.TensorShapeProto.newBuilder();
            Matcher m = p.matcher(layerList.get(i).getLayerShape());
            int dim_index = 0;
            int size = 1;
            while (m.find()) {
                int dim = Integer.parseInt(m.group());
                size = size * dim;
                TensorEntity.TensorShapeProto.Dim.Builder dimBuilder =
                        TensorEntity.TensorShapeProto.Dim.newBuilder();
                dimBuilder.setSize(dim);
                tensorShapeBuilder.addDim(dim_index, dimBuilder);
                dim_index++;
            }
            FloatBuffer floatBuffer = FloatBuffer.allocate(size);
            Tensor weights = session.runner().
                    fetch(layerList.get(i).getLayerName()).run().get(0);
            weights.writeTo(floatBuffer);
            float[] floats = floatBuffer.array();
            for (float aFloat : floats) {
                tensorBuilder.addFloatVal(aFloat);
            }
            tensorBuilder.setTensorShape(tensorShapeBuilder);
            modelWeightsBuilder.addTensor(i, tensorBuilder);
        }
        return modelWeightsBuilder;
    }
}
