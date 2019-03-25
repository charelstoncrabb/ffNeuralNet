#pragma once

#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

#include "ActivationFunctions.h"

//#define DEBUG

//! Simple feed-forward network with single activation function at all levels
class FFNeuralNet
{
public:
	FFNeuralNet(int nInputNodes = 1, int nHiddenNodes = 1, int nHiddenLayers = 1, int nOutputNodes = 1,
		double rate = 0.2,
		std::pair<std::function<Eigen::MatrixXd(Eigen::MatrixXd)>,
		std::function<Eigen::MatrixXd(Eigen::MatrixXd)>> ACTFUN =
		std::pair<std::function<Eigen::MatrixXd(Eigen::MatrixXd)>,
		std::function<Eigen::MatrixXd(Eigen::MatrixXd)>>(actFuns::sigmoid, actFuns::sigmoid_prime));



	void train(Eigen::MatrixXd* X, Eigen::MatrixXd* Y, double tol);

	double test(Eigen::MatrixXd* X, Eigen::MatrixXd* Y);

	Eigen::MatrixXd predict(Eigen::MatrixXd);

private:
	double backPropGradient(const Eigen::MatrixXd Xi, const Eigen::MatrixXd Yi);


	const int nInputNodes;
	const int nHiddenNodes;
	const int nHiddenLayers;
	const int nOutputNodes;
	const int netDepth;
	double rate;

	// Layer Activation Function & Derivative:
	std::function<Eigen::MatrixXd(Eigen::MatrixXd)> actFun;
	std::function<Eigen::MatrixXd(Eigen::MatrixXd)> actFun_prime;

	// Network Parameters:
	std::vector<Eigen::MatrixXd> weights;
	std::vector<Eigen::MatrixXd> biases;
	std::vector<Eigen::MatrixXd> activations;

	std::vector<Eigen::MatrixXd> prev_weights;
	std::vector<Eigen::MatrixXd> prev_biases;

	// Containers for Cost gradient wrt respective Network Parameters
	std::vector<Eigen::MatrixXd> grad_weights;
	std::vector<Eigen::MatrixXd> grad_biases;
	std::vector<Eigen::MatrixXd> grad_activations;

	std::vector<Eigen::MatrixXd> prev_grad_weights;
	std::vector<Eigen::MatrixXd> prev_grad_biases;

	// Containers for training data
	Eigen::MatrixXd *train_X;
	Eigen::MatrixXd *train_Y;

	friend class FFNNTest;
};
