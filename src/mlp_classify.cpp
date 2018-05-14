#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include <Eigen/Dense>
#include "../include/neural_networks/MultiLayerPerceptron.hpp"
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
  cout << "mlp-classify [configFile] [validationDataFile] [labelsDataFile] [savedWeightsFile]\n";
}

int main(int argc, char **argv) {
  if(argc != 5) {
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

  printf("Loading validation data from %s...\n", argv[2]);
  vector< vector<double> > validationData = utils::Misc::fetchData(argv[2]);

  printf("Loading validation labels from %s...\n", argv[3]);
  vector< vector<double> > labelsData = utils::Misc::fetchData(argv[3]);

  printf("Loading weights from %s...\n", argv[4]);
  mlp->loadWeights(argv[4]);

  for(int i = 0; i < validationData.size(); i++) {
    mlp->setInput(validationData.at(i)); 
    mlp->feedForward();

    printf("Output for datapoint %d:\n", i);
    cout << mlp->layers.back() << endl;

    MatrixXd target(1, labelsData.at(i).size());
    for(int j = 0; j < labelsData.at(i).size(); j++) {
      target(0, j) = labelsData.at(i).at(j);
    }

    printf("Target for datapoint %d:\n", i);
    cout << target << endl;

    printf("======================\n");
    //printf("%f\n", mlp->loss);
  }

  printf("Done...\n");

  return 0;
}
