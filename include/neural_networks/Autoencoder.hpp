#ifndef _AUTOENCODER_HPP_
#define _AUTOENCODER_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <iomanip>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "./MultiLayerPerceptron.hpp"
#include "../json.hpp"

using namespace std;
using namespace cv;
using Eigen::MatrixXd;
using json = nlohmann::json;

class Autoencoder : public MultiLayerPerceptron
{
public:
  Autoencoder(Config config) : MultiLayerPerceptron(config) {
    if(config.imageShape.size() == 2) {
      imageShape    = config.imageShape;
    }
  };

private:
  vector<int> imageShape;
  Mat currentImage;
  string windowName = "Result";
};

#endif
