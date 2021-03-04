import tensorflow as tf


def write_graph_netInfo(args, sess, net_info, scope_name):
    tf.compat.v1.train.write_graph(sess.graph, args.graph_path, args.model_name + ".pb", as_text=False)

    # generate net info
    trainable_var = tf.trainable_variables()
    global_var = tf.global_variables()
    with open(args.graph_path + "/" + args.model_name + "_trainable_var" + ".txt", "w") as f:
        variables_sum = 0
        for t_var in trainable_var:
            accumulate = 1
            for i in range(len(t_var.shape)):
                accumulate = t_var.shape[i] * accumulate
            print(t_var.shape, t_var.initial_value.op.name, ":", accumulate)
            variables_sum = accumulate + variables_sum
            if scope_name and scope_name in t_var.initial_value.op.name:
                f.write(t_var.initial_value.op.name + ";" + str(t_var.op.name) + "\n")
            elif scope_name is None:
                f.write(t_var.initial_value.op.name + ";" + str(t_var.op.name) + "\n")
            else:
                continue
        print(variables_sum)

    with open(args.graph_path + "/" + args.model_name + "_global_var" + ".txt", "w") as f:
        variables_sum = 0
        for t_var in global_var:
            accumulate = 1
            for i in range(len(t_var.shape)):
                accumulate = t_var.shape[i] * accumulate
            variables_sum = accumulate + variables_sum
            if scope_name and scope_name in t_var.initial_value.op.name:
                f.write(t_var.initial_value.op.name + ";" + str(t_var.shape) + "\n")
            elif scope_name is None:
                f.write(t_var.initial_value.op.name + ";" + str(t_var.shape) + "\n")
            else:
                continue
        print(variables_sum)

    with open(args.graph_path + "/" + args.model_name + "_train_info" + ".txt", "w") as f:
        f.write(net_info["y"].op.name + ";" + str(net_info["y"].shape) + "\n")
        f.write(net_info["x"].op.name + ";" + str(net_info["x"].shape) + "\n")
        f.write(net_info['global_var_init'].name + ";" + "---" + "\n")
        f.write(net_info["train_op"].name + ";" + "---" + "\n")
        f.write(net_info["loss_op"].name + ";" + "---" + "\n")
        f.write(net_info["accuracy_op"].name + ";" + "---" + "\n")