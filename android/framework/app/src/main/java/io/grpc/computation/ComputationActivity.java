/*
 * Copyright 2015 The gRPC Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.grpc.computation;

import android.content.Context;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.UUID;

public class ComputationActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private Button trainButton;
    private TextView resultText;
    private EditText round;
    private Spinner data;
    private Spinner model;
    private static final String DATA_FILE = "file:///android_asset/bank_zhongyuan/test_data1.csv";
    private Context context;
    private static String local_id = UUID.randomUUID().toString().replaceAll("-", "");
    private static String placeholder;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_computation);
        trainButton = (Button) findViewById(R.id.train_button);
        round = (EditText) findViewById(R.id.round);
        data = (Spinner) findViewById(R.id.data);
        model = (Spinner) findViewById(R.id.model);
        resultText = (TextView) findViewById(R.id.server_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
        this.context = getApplicationContext();
    }

    /**
     *
     * @param view
     */
    public void Training(View view) {
        trainButton.setEnabled(false);
        resultText.setText("");
        new TrainingTask.LocalTrainingTask(this, this.context, this.resultText).execute(
                local_id,
                (String) data.getSelectedItem(),
                (String) model.getSelectedItem(),
                placeholder,
                round.getText().toString()
                );
    }


    /**
     * this is a test example of mul
     * @param view
     */
    /*
    public void Mul(View view) {
        sendButton.setEnabled(false);
        resultText.setText("");
        new ArithmeticTask.MulTask(this)
                .execute(
                        hostEdit.getText().toString(),
                        portEdit.getText().toString(),
                        "5",
                        "10"
                );
    }
    *
     */
}
