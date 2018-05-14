#include "../../../include/neural_networks/Autoencoder.hpp"

MatrixXd Autoencoder::vectorToImage(MatrixXd m) {
  MatrixXd result = MatrixXd::Zero(imageShape[0], imageShape[1]);

  for(int d = 0; d < m.rows(); d++) {
    for(int r = 0; r < imageShape[0]; r++) {
      for(int c = 0; c < imageShape[1]; c++) {
        result(r, c) = m(d, 0);
      }
    }
  }

  return result;
}

MatrixXd Autoencoder::vectorToImage(vector<double> v) {
  MatrixXd result = MatrixXd::Zero(imageShape[0], imageShape[1]);

  for(int d = 0; d < v.size(); d++) {
    for(int r = 0; r < imageShape[0]; r++) {
      for(int c = 0; c < imageShape[1]; c++) {
        result(r, c) = v.at(d);
      }
    }
  }

  return result;
}
