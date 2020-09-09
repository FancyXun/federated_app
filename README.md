Federated Learning Application
==============================================

You may want to read through the
[Quick Start](https://grpc.io/docs/languages/java/quickstart)
before trying out this repo.


### <a name="Build"></a> Build

1. Install server and client in PC:
```
$ ./gradlew installDist
```

This creates the scripts `computation-server`, `computation-client`. in the
`build/install/learning/bin/` directory that run the example. The
example requires the server to be running before starting the client.

First run server:

```
$ ./build/install/federated_learning/bin/computation-server
```

And in a different terminal window run:

```
$ cd android/framework; ../../gradlew installDebug
```

Launch the client app from your device.


Click “MUL” . You’ll see multiplication of x and y

That's it!

For more information, refer to gRPC Java's [tutorial](https://grpc.io/docs/languages/java/basics).
