#include "ActFunTest.h"



ActFunTest::ActFunTest()
{
}


ActFunTest::~ActFunTest()
{
}

bool ActFunTest::xany(std::vector<bool> a)
{
	for (auto r : a)
		if (r)
			return false;
	return true;
}

void 
ActFunTest::RunTests(void)
{
	std::vector<bool> testResults;

	testResults.push_back(runReLUTest());
	testResults.push_back(runSigmoidTest());
	// Add more tests by adding to testResults vector here

	if (xany(testResults))
		std::cout << "ActFunTest all tests pass" << std::endl;
}

bool 
ActFunTest::runReLUTest(void)
{
	Eigen::MatrixXd mat(3,1), out(3,1), ans(3,1);

	mat << -10.0, 0.0, 10.0;
	ans << 0.0, 0.0, 10.0;

	out = actFuns::ReLU(mat);

	if (((out - ans).transpose()*(out - ans)).trace() > 1.e-12)
	{
		std::cout << "runReLUTest Failure!" << std::endl;
		return true;
	}

	ans(2, 0) = 1.0;
	out = actFuns::ReLU_prime(mat);

	if (((out - ans).transpose()*(out - ans)).trace() > 1.e-12)
	{
		std::cout << "runReLUTest Failure!" << std::endl;
		return true;
	}

	return false;
}

bool ActFunTest::runSigmoidTest(void)
{
	Eigen::MatrixXd mat(1, 1), out(1, 1), ans(1, 1);
	mat << 0.0;
	ans << 0.5;

	out = actFuns::sigmoid(mat);

	if (((out - ans).transpose()*(out - ans)).trace() > 1.e-12)
	{
		std::cout << "runSigmoidTest Failure!" << std::endl;
		return true;
	}

	ans(0, 0) = 0.25;
	out = actFuns::sigmoid_prime(mat);

	if (((out - ans).transpose()*(out - ans)).trace() > 1.e-12)
	{
		std::cout << "runSigmoidTest Failure!" << std::endl;
		return true;
	}

	return false;
}