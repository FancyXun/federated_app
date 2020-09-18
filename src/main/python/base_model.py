import argparse
import json
import time

from google.protobuf import json_format
import tensorflow as tf

parse = argparse.ArgumentParser()
parse.add_argument('-p', '--path', help='the graph root path ')
parse.add_argument('-g_name', '--graph_name', help='the graph name')

args = vars(parse.parse_args())
path = args["path"]
graph_name = args["graph_name"]


class Model(object):
    def __init__(self):
        self.graph = None
        self.var = []
        self.placeholder_name = []
        self.const = []

    def assign_var(self, target_var):
        assert len(self.var) == len(target_var)
        assert self.graph is not None
        bp_target = {}
        for i, j in zip(self.var, target_var):
            name = i.op.name + "_" + str(time.time()).replace(".", "")
            i.assign(j, name=name)
            bp_target[name] = {}
            bp_target[name]['shape'] = [str(i.value) for i in list(i.shape)]
        self.write_json(bp_target)
        self.write_pb()

    def write_json(self, assign_target):
        graph_def = self.graph.as_graph_def()
        json_string = json_format.MessageToJson(graph_def)
        obj = json.loads(json_string)
        var = {}
        init_var_target = {}
        for i in obj['node']:
            if i['op'] == 'Assign':
                init_var_target[i['name']] = i['input']
            elif i['op'] in ("Variable", "VariableV2", "VarHandleOp"):
                var[i['name']] = i['attr']['shape']
            elif i['op'] == 'Const':
                self.const.append(i['name'])
                var[i['name']] = i['attr']['value']['tensor']['tensorShape']
        for key, val in init_var_target.items():
            for v in val:
                if v in self.const:
                    init_var_target[key] = {"parentNode": v,
                                            "shape": [i['size'] for i in var[v]['dim']]}
                    break
        for i in assign_target.keys():
            init_var_target.pop(i)
        var_name = {
            "placeholder": self.placeholder_name,
            "varTarget": init_var_target,
            "assignTarget": assign_target
        }
        with open(path + "/" + graph_name.split(".")[0] + ".json", "w", encoding='utf-8') as f:
            json.dump(var_name, f, indent=2, sort_keys=True, ensure_ascii=False)

    def write_pb(self):
        tf.compat.v1.train.write_graph(self.graph, path, graph_name, as_text=False)
