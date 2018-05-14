#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::activate(MatrixXd &m, ACTIVATION activation) {
  int numRows = m.rows();
  int numCols = m.cols();

  switch(activation) {
    case A_RELU:
      for(int r = 0; r < numRows; r++) {
        for(int c = 0; c < numCols; c++) {
          if(m(r, c) < 0) {
            m(r, c) = 0.00;
          }
        }
      }

      break;
    case A_SIGM:
      for(int r = 0; r < numRows; r++) {
        for(int c = 0; c < numCols; c++) {
          m(r, c) = 1 / (1 + exp(-m(r, c)));
        }
      }

      break;
  };
}
