#include "../../include/nn_utils/Misc.hpp"

vector< vector<double> > nn_utils::Misc::fetchData(string path) {
  vector< vector<double> > data;

  ifstream infile(path);

  string line;
  while(getline(infile, line)) {
    vector<double>  dRow;
    string          tok;
    stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      double s = stod(tok);
      dRow.push_back(stod(tok));
    }

    data.push_back(dRow);
  }

  return data;
}
