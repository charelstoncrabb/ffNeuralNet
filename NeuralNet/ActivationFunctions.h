#pragma once
#include <Eigen/Dense>
#include <utility>

namespace actFuns
{
	Eigen::MatrixXd ReLU(Eigen::MatrixXd mat);

	Eigen::MatrixXd ReLU_prime(Eigen::MatrixXd mat);

	Eigen::MatrixXd ReQU(Eigen::MatrixXd mat);

	Eigen::MatrixXd ReQU_prime(Eigen::MatrixXd mat);

	Eigen::MatrixXd sigmoid(Eigen::MatrixXd mat);

	Eigen::MatrixXd sigmoid_prime(Eigen::MatrixXd mat);
}

#define SIGMOID std::pair< std::function<Eigen::MatrixXd(Eigen::MatrixXd)>, std::function<Eigen::MatrixXd(Eigen::MatrixXd)> >(actFuns::sigmoid,actFuns::sigmoid_prime) 

#define RELU std::pair< std::function<Eigen::MatrixXd(Eigen::MatrixXd)>, std::function<Eigen::MatrixXd(Eigen::MatrixXd)> >(actFuns::ReLU,actFuns::ReLU_prime) 

#define REQU std::pair< std::function<Eigen::MatrixXd(Eigen::MatrixXd)>, std::function<Eigen::MatrixXd(Eigen::MatrixXd)> >(actFuns::ReQU,actFuns::ReQU_prime) 
