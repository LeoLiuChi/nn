#ifndef _MULTI_LAYER_PERCEPTRON_HPP_
#define _MULTI_LAYER_PERCEPTRON_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <iomanip>
#include <Eigen/Dense>

#include "../json.hpp"

using namespace std;
using Eigen::MatrixXd;

using json = nlohmann::json;

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
  vector<int> imageShape;
};

class MultiLayerPerceptron
{
public:

  Config config;
  vector<MatrixXd> weights;
  vector<MatrixXd> biasWeights;
  vector<MatrixXd> deltaWeights;
  vector<MatrixXd> layers;
  vector<MatrixXd> gradients;

  MatrixXd target;
  MatrixXd errors;
  MatrixXd eDerivatives;

  double loss = 0.00;

  /*
   * CORE METHODS
   */
  void feedForward();
  void backProp();
  void activate(MatrixXd &m, ACTIVATION activation);
  void calculateLoss(vector<double> target);
  void loadWeights(string filename);
  void saveWeights(string filename);

  void setInput(vector<double> input) {
    MatrixXd buff(input.size(), 1);

    for(int i = 0; i < input.size(); i++) {
      buff(i, 0) = input.at(i);
    }

    this->layers[0] = buff;
  };

  void setTarget(vector<double> t) {
    MatrixXd buff(t.size(), 1);

    for(int i = 0; i < t.size(); i++) {
      buff(0, i) = t.at(i);
    }

    target  = buff;
  };

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

    cout << "Weights: ";
    for(int i = 0; i < weights.size(); i++) {
      cout << i << ": " << weights.at(i).rows() << "x" << weights.at(i).cols() << endl;
    }

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
  MatrixXd derive(MatrixXd, ACTIVATION);
  void initWeights(double min = -1.0, double max = 1.0, int seed = 1.0);
  void initTopology();
};

#endif
