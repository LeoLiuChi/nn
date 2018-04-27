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
  cout << "mlp_train [configFile]\n";
}

int main(int argc, char **argv) {
  if(argc != 2) {
    print_syntax();
    exit(-1);
  }

  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  MultiLayerPerceptron *mlp = new MultiLayerPerceptron(
                                build_config_from_json(json::parse(str))
                              );

  mlp->printConfig();
  mlp->printWeights();

  /*

  MatrixXf m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  MatrixXf n(1, 2);
  n(0,0) = 3;
  n(0,1) = 2.5;
  cout << "Rows: " << m.rows() << " Cols: " << m.cols() << endl;
  cout << "Rows: " << n.rows() << " Cols: " << n.cols() << endl;

  MatrixXf z = (n * m);
  MatrixXf b(1, z.cols());

  cout << z << endl;
  cout << (z.array() + 1).matrix() << endl;
  cout << (m.array() * m.array()).matrix() << endl;
  */
  return 0;
}
