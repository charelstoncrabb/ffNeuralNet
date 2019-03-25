#include "ActivationFunctions.h"

Eigen::MatrixXd actFuns::ReLU(Eigen::MatrixXd mat)
{
	return mat.cwiseMax(0.0);
}

Eigen::MatrixXd actFuns::ReLU_prime(Eigen::MatrixXd mat)
{
	for (int i = 0; i < mat.rows(); i++)
		for (int j = 0; j < mat.cols(); j++)
			mat(i, j) = mat(i, j) > 0. ? 1 : 0.;
	return mat;
}

Eigen::MatrixXd actFuns::ReQU(Eigen::MatrixXd mat)
{
	mat = mat.cwiseMax(0.0);
	mat = mat.cwiseProduct(mat);
	return mat;
}

Eigen::MatrixXd actFuns::ReQU_prime(Eigen::MatrixXd mat)
{
	return ReLU(mat).array() * 2;
}

Eigen::MatrixXd actFuns::sigmoid(Eigen::MatrixXd mat)
{
	return mat.array().exp() / (mat.array().exp() + 1);
}

Eigen::MatrixXd actFuns::sigmoid_prime(Eigen::MatrixXd mat)
{
	return sigmoid(mat).array()*(1 - sigmoid(mat).array());
}