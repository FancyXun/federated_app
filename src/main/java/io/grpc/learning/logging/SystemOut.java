package io.grpc.learning.logging;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;



public class SystemOut {

    private volatile static SystemOut instance = null;
    PrintStream ps;

    {
        try {
            ps = new PrintStream(new FileOutputStream("logging.txt", true));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static SystemOut getInstance() {
        if (instance == null) {
            synchronized (SystemOut.class) {
                if (instance == null) {
                    instance = new SystemOut();
                }
            }

        }
        return instance;
    }
    public void output(String text, PrintStream ps1) {
        ps1.println(text);
        ps.println(text);
    }
}
