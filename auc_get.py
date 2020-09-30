import tensorflow as tf

graph = tf.Graph()
nth = 1000

def auc_tf(lab,pre,num_thresholds=1000):
    auc_value, auc_op = tf.metrics.auc(lab, pre, num_thresholds=num_thresholds,name='auc_pair')
    return auc_op

with graph.as_default():
    #auc_test
    labels = tf.placeholder(shape=(None,), dtype=tf.float32,name='labels')
    predictions = tf.placeholder(shape=(None,), dtype=tf.float32,name='predictions')
    auc = auc_tf(labels,predictions,nth)
    local_var = tf.local_variables()
    init_new_vars_op = tf.initialize_variables(local_var)

tf.compat.v1.train.write_graph(graph,'./','getauc.pb',as_text=False)

""" java code for calculating auc with a given labels_predictions pair

        float labarr [] = {0,0,0,0,0,0,1,1,1,1,1,1};
        float prearr [] = {0.3f,0.4f,0.6f,0.2f,0.7f,0.5f,0.1f,0.9f,0.7f,0.5f,0.9f,0.8f};
        float init [] = new float [1000];
        System.out.println(init);
        Tensor x = Tensor.create(labarr);
        Tensor y = Tensor.create(prearr);
        System.out.println("tensor,ok! :"+x+y);
        System.out.println("array,ok!" + labarr + prearr);

        s.runner().feed("auc_pair/true_positives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/true_positives/Assign").run();
        s.runner().feed("auc_pair/false_positives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/false_positives/Assign").run();
        s.runner().feed("auc_pair/true_negatives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/true_negatives/Assign").run();
        s.runner().feed("auc_pair/false_negatives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/false_negatives/Assign").run();
        Tensor z = s.runner().feed("labels", x).feed("predictions",y).fetch("auc_pair/update_op").run().get(0);

        System.out.println("AUC: "+z.floatValue());
        System.out.println("to the end");
"""
