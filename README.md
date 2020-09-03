gRPC Simulation
==============================================

You may want to read through the
[Quick Start](https://grpc.io/docs/languages/java/quickstart)
before trying out the examples.

## Basic examples

- [Hello world](src/main/java/io/grpc/learning/helloworld)


### <a name="to-build-the-examples"></a> To build the examples

1. From federated_mobile/simulation directory:
```
$ ./gradlew installDist
```

This creates the scripts `hello-world-server`, `hello-world-client`. in the
`build/install/learning/bin/` directory that run the examples. Each
example requires the server to be running before starting the client.

For example, to try the hello world example first run:

```
$ ./build/install/learning/bin/hello-world-server
```

And in a different terminal window run:

```
$ cd android/helloworld; ../../gradlew installDebug
```

Launch the client app from your device.

In the client app, enter the server’s Host and Port information. The values you enter depend on the device kind (real or virtual) — for details, see [Connecting to the server](https://grpc.io/docs/languages/android/quickstart/#connecting-to-the-server) below.

Type “Alice” in the Message box and click Send. You’ll see the following response:
```
Hello Alice
```

That's it!

For more information, refer to gRPC Java's [README](../README.md) and
[tutorial](https://grpc.io/docs/languages/java/basics).
