#pragma once

#include <vector>
#include <iostream>
#include "FFNeuralNet.h"
#include "MNISTFileParser.h"
#include "ActivationFunctions.h"

class FFNNTest
{
public:
	FFNNTest();
	~FFNNTest();

	void RunTests(void);
	bool xany(std::vector<bool> a);

	bool testNot(void);
	bool testIdRegression(void);
	bool testMNIST(int train_record_cap = -1, int test_record_cap = -1);
};

