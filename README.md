# NN

Neural Network based models written in C++, Eigen and OpenCV

# Requirements

* CMake 3
* Eigen3
* OpenCV 3.x

# Installation

```
$ cmake .
$ make
```

# Components

## MLP: Multilayer Perceptron

A typical fully connected multilayer neural network

### Config File

Json file with the following parameters:

```
{
  "topology": [4, 3, 1],
  "bias": 0.01,
  "learningRate": 0.001,
  "momentum": 1,
  "epoch": 10000,
  "hActivation": 1,
  "oActivation": 2,
  "cost": 0
}
```

### Activation Values

* 0 - TANH
* 1 - RELU
* 2 - SIGMOID

### Cost Values

* 0 - MSE (Least Squares Error)

### MLP Training (mlp-train)

Saves a json weight array in file [weightFile.json]

```
./mlp-train [configFile.json] [trainingData.csv] [trainingLabels.json] [weightFile.json]
```

### MLP Classification (mlp-classify)

Classifies validation data given set of labels given saved weightFile.json

```
./mlp-classify [configFile.json] [validationData.csv] [validationLabels.json] [weightFile.json]
```
