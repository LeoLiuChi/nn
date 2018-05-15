#ifndef _AUTOENCODER_HPP_
#define _AUTOENCODER_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <iomanip>
#include <Eigen/Dense>

#include "./MultiLayerPerceptron.hpp"
#include "../json.hpp"

using namespace std;
using Eigen::MatrixXd;
using json = nlohmann::json;

class Autoencoder : public MultiLayerPerceptron
{
public:
  Autoencoder(Config config) : MultiLayerPerceptron(config) {
    if(config.imageShape.size() == 2) {
      imageShape          = config.imageShape;
    }
  };

  vector<int> imageShape;

private:
};

#endif
