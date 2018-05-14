#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include <Eigen/Dense>
#include "../include/neural_networks/MultiLayerPerceptron.hpp"
#include "../include/neural_networks/Autoencoder.hpp"
#include "../include/json.hpp"
#include "../include/utils/Misc.hpp"

using namespace std;
using json = nlohmann::json;
using Eigen::MatrixXf;

Config build_config_from_json(json o) {
  Config c;

  vector<int> topology    = o["topology"];
  double bias             = o["bias"];
  double learningRate     = o["learningRate"];
  double momentum         = o["momentum"];
  int epoch               = o["epoch"];
  ACTIVATION hActivation  = o["hActivation"];
  ACTIVATION oActivation  = o["oActivation"];
  COST cost               = o["cost"];

  c.topology      = topology;
  c.bias          = bias;
  c.learningRate  = learningRate;
  c.momentum      = momentum;
  c.epoch         = epoch;
  c.hActivation   = hActivation;
  c.oActivation   = oActivation;
  c.cost          = cost;

  return c;
};

void print_syntax() {
  cout << "Syntax:\n";
  cout << "ae-train [configFile] [trainingDataFile] [savedWeightsFile]\n";
}

int main(int argc, char **argv) {
  if(argc != 4) {
    print_syntax();
    exit(-1);
  }

  printf("Loading configuration file from %s...\n", argv[1]);
  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  MultiLayerPerceptron *mlp = new MultiLayerPerceptron(
                                build_config_from_json(json::parse(str))
                              );

  printf("Neural network initiated...");
  mlp->printConfig();

  printf("Loading data file from %s...\n", argv[2]);
  vector< vector<double> > trainingData = utils::Misc::fetchData(argv[2]);

  printf("Loading labels file from %s...\n", argv[3]);
  vector< vector<double> > labelsData = utils::Misc::fetchData(argv[2]);

  double err = 0.00;

  for(int i = 0; i < mlp->config.epoch; i++) {
    double aveLoss  = 0.00;
    for(int j = 0; j < trainingData.size(); j++) {
      mlp->setInput(trainingData.at(j)); 
      mlp->feedForward();
      mlp->calculateLoss(labelsData.at(j));
      mlp->backProp();
      aveLoss += mlp->loss;
    }

    aveLoss = aveLoss / trainingData.size();
    printf("Loss: %f\n", aveLoss);
  }

  printf("Saving weights to %s...\n", argv[3]);
  mlp->saveWeights(argv[3]);

  printf("Done...\n");

  return 0;
}
