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
This would give you log info like bellow
```
{Datetime} io.grpc.learning.computation.ComputationServer start
INFO: Server started, ip is {server ip} listening on 50051
```

And in a different terminal window run:

```
$ cd android/framework; ../../gradlew installDebug
```

Launch the client app from your device.


Input server ip and local epoch number, Click “联邦训练” . You’ll see the loss of Logistic regression

That's it!

##### Note
The client (android device) can access the server.

For more information, refer to gRPC Java's [tutorial](https://grpc.io/docs/languages/java/basics).
