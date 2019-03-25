#include "FFNeuralNet.h"
#include "ActivationFunctions.h"
#include "MNISTFileParser.h"
#include "FFNNTest.h"
#include "ActFunTest.h"

#include <iostream>
#include <string>


int main()
{
	ActFunTest aft;
	aft.RunTests();

	FFNNTest nntests;

	nntests.testMNIST(10000, 1000);
	

	std::cin.get();
	return 0;
}
