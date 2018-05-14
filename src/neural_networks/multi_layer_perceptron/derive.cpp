#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

MatrixXd MultiLayerPerceptron::derive(MatrixXd m, ACTIVATION a) {
  int numRows = m.rows();
  int numCols = m.cols();

  MatrixXd buff = MatrixXd::Zero(m.rows(), m.cols());

  switch(a) {
    case A_RELU:
      for(int r = 0; r < numRows; r++) {
        for(int c = 0; c < numCols; c++) {
          if(m(r, c) > 0) {
            buff(r, c) = 1.0;
          } else {
            buff(r, c) = 0.0;
          }
        }
      }

      break;
    case A_SIGM:
      buff = (m.array() * (1 - m.array())).matrix();
      
      break;
  };

  return buff;
};
