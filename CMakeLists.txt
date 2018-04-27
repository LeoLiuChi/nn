cmake_minimum_required(VERSION 3.7)
project(nn)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(
  mlp_train
  src/mlp_train.cpp
  src/neural_networks/multi_layer_perceptron/setup.cpp
  src/neural_networks/multi_layer_perceptron/init_weights.cpp
)

target_link_libraries(mlp_train Eigen3::Eigen)
