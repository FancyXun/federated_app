package io.grpc.learning.utils;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.grpc.learning.vo.TensorVarName;

public class JsonUtils {

    private static String assignTarget = "assignTarget";
    private static String varTarget = "varTarget";
    private static String placeholder = "placeholder";
    private static String parentNode = "parentNode";
    private static String shape = "shape";

    public static String readJsonFile(String fileName) {
        String jsonStr = "";
        try {
            File jsonFile = new File(fileName);
            FileReader fileReader = new FileReader(jsonFile);
            Reader reader = new InputStreamReader(new FileInputStream(jsonFile), "utf-8");
            int ch;
            StringBuffer sb = new StringBuffer();
            while ((ch = reader.read()) != -1) {
                sb.append((char) ch);
            }
            fileReader.close();
            reader.close();
            jsonStr = sb.toString();
            return jsonStr;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static TensorVarName jsonToMap(JSONObject jsonObject) {
        List<String> tensorName = new ArrayList<>();
        List<String> placeholderName = new ArrayList<>();
        List<String> tensorTargetName = new ArrayList<>();
        List<List<Integer>> tensorShape = new ArrayList<>();
        List<String> tensorAssignName = new ArrayList<>();
        TensorVarName tensorVarName = new TensorVarName();
        JSONArray jsonArray = jsonObject.getJSONArray(assignTarget);
        JSONArray jsonArray1 = jsonObject.getJSONArray(placeholder);
        for (int i = 0; i < jsonArray1.size(); i++) {
            placeholderName.add((String) jsonArray1.get(i));
        }
        for (int i = 0; i < jsonArray.size(); i++) {
            tensorAssignName.add((String) jsonArray.get(i));
        }
        Set<Map.Entry<String, Object>> entrySet = jsonObject.getJSONObject(varTarget).entrySet();
        for (Iterator<Map.Entry<String, Object>> it = entrySet.iterator(); it.hasNext(); ) {
            Map.Entry<String, Object> var = it.next();
            List<Integer> integerList = new ArrayList<>();
            tensorTargetName.add(var.getKey());
            JSONObject obj = (JSONObject) var.getValue();
            tensorName.add((String) obj.get(parentNode));
            JSONArray shapeList = obj.getJSONArray(shape);
            for (int i = 0; i < shapeList.size(); i++) {
                integerList.add(Integer.parseInt((String) shapeList.get(i)));
            }
            tensorShape.add(integerList);
        }
        tensorVarName.setTensorName(tensorName);
        tensorVarName.setTensorTargetName(tensorTargetName);
        tensorVarName.setTensorShape(tensorShape);
        tensorVarName.setPlaceholder(placeholderName);
        tensorVarName.setTensorAssignName(tensorAssignName);
        return tensorVarName;
    }
}
