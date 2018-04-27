#ifndef _MULTI_LAYER_PERCEPTRON_HPP_
#define _MULTI_LAYER_PERCEPTRON_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

enum ACTIVATION {
  A_TANH,
  A_RELU,
  A_SIGM
};

enum COST {
  MSE
};

struct Config {
  vector<int> topology;
  double bias;
  double learningRate;
  double momentum;
  int epoch;
  ACTIVATION hActivation;
  ACTIVATION oActivation;
  COST cost;
};

class MultiLayerPerceptron
{
public:

  Config config;
  vector<MatrixXd> weights;

  MultiLayerPerceptron(Config config) {
    this->config  = config;
    this->setup();
  };

  void printWeights() {
    cout << "Weights for MLP\n";
    cout << "====================\n";
    for(int i = 0; i < config.topology.size() - 1; i++) {
      cout << "Weight: " << i << "\n";
      cout << weights.at(i) << "\n";
      cout << "--------------------\n";
    }
  };

  void printConfig() {
    cout << "Config for MLP\n";
    cout << "====================\n";

    cout << "Topology: ";
    for(int i = 0; i < config.topology.size(); i++) {
      cout << config.topology.at(i) << "\t";
    }

    cout << "\n";

    cout << "Bias: " << config.bias << "\n";
    cout << "Learning Rate: " << config.learningRate << "\n";
    cout << "Momentum: " << config.momentum << "\n";
    cout << "Epoch: " << config.epoch << "\n";
    cout << "Hidden Layer Activation: ";

    switch(config.hActivation) {
      case A_TANH:
        cout << "TANH";
        break;
      case A_SIGM:
        cout << "SIGM";
        break;
      case A_RELU:
        cout << "RELU";
        break;
    }

    cout << "\n";
  };

private:
  void setup();
  void initWeights(double min = -1.0, double max = 1.0);
};

#endif
