#pragma once
#include "ActivationFunctions.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

class ActFunTest
{
public:
	ActFunTest();
	~ActFunTest();

	void RunTests(void);

	bool runReLUTest(void);

	bool runSigmoidTest(void);

	bool xany(std::vector<bool>);
};

