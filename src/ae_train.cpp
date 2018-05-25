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
#include "../include/nn_utils/Misc.hpp"

using namespace std;
using json  = nlohmann::json;
using Eigen::MatrixXf;

Config build_ae_config_from_json(json o) {
  Config c;

  // Symmetrical topology
  vector<int> tempTopology  = o["topology"];
  vector<int> topology;
  for(int i = 0; i < tempTopology.size(); i++) {
    topology.push_back(tempTopology.at(i));
  }

  for(int i = tempTopology.size() - 2; i >= 0; i--) {
    topology.push_back(tempTopology.at(i));
  }

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


  Autoencoder *ae = new Autoencoder(
                      build_ae_config_from_json(json::parse(str))
                    );

  printf("Neural network initiated...");
  ae->printConfig();

  printf("Loading data file from %s...\n", argv[2]);
  vector< vector<double> > trainingData = nn_utils::Misc::fetchData(argv[2]);

  printf("Loading labels file from %s...\n", argv[2]);
  vector< vector<double> > labelsData = nn_utils::Misc::fetchData(argv[2]);

  double err = 0.00;

  for(int i = 0; i < ae->config.epoch; i++) {
    double aveLoss  = 0.00;
    for(int j = 0; j < trainingData.size(); j++) {
      ae->setInput(trainingData.at(j)); 
      ae->feedForward();
      ae->calculateLoss(labelsData.at(j));
      ae->backProp();
      aveLoss += ae->loss;
    }

    aveLoss = aveLoss / trainingData.size();
    printf("Epoch %d, Loss: %f\n", i, aveLoss);
  }

  printf("Saving weights to %s...\n", argv[3]);
  ae->saveWeights(argv[3]);

  printf("Done...\n");

  return 0;
}
