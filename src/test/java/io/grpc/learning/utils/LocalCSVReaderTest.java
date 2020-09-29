package io.grpc.learning.utils;

import org.junit.Test;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LocalCSVReaderTest {

    @Test
    public void PatternTest() {
        String line = "train@1-7";
        String pattern = "(.*@)(([0-9]+)-([0-9]+))*";

        Pattern r = Pattern.compile(pattern);

        Matcher m = r.matcher(line);
        if (m.find( )) {
            System.out.println("Found value: " + m.group(1) );
            System.out.println("Found value: " + m.group(2) );
            System.out.println("Found value: " + m.group(3) );
            System.out.println("Found value: " + m.group(4) );
        } else {
            System.out.println("NO MATCH");
        }

    }
}
