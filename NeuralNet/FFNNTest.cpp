#include "FFNNTest.h"



FFNNTest::FFNNTest()
{
}


FFNNTest::~FFNNTest()
{
}

bool FFNNTest::xany(std::vector<bool> a)
{
	for (auto r : a)
		if (r)
			return false;
	return true;
}


void
FFNNTest::RunTests(void)
{
	std::vector<bool> testResults;

	testResults.push_back(testNot());
	testResults.push_back(testIdRegression());
	testResults.push_back(testMNIST());
	// Add more tests by adding to testResults vector here

	if (xany(testResults))
		std::cout << "ActFunTest all tests pass" << std::endl;
}


bool 
FFNNTest::testNot(void)
{
	std::cout << "Testing NOT operation..." << std::endl;
	FFNeuralNet knot(1, 2, 1, 1, 0.2);

	Eigen::MatrixXd train_x(2, 1), train_y(2, 1);
	train_x << 1., 0.;
	train_y << 0., 1.;

	knot.train(&train_x, &train_y, 1.e-6);

	return false;
}


bool
FFNNTest::testIdRegression(void)
{
	int height = 1, depth = 1, nNodes = 2;

	FFNeuralNet id(height, nNodes, depth, height, 0.2);

	Eigen::MatrixXd X, Y;
	X = Eigen::MatrixXd::Random(height, 10000).cwiseAbs();
	Y = X;

	id.train(&X, &Y, 1.e-6);

	return false;
}


bool
FFNNTest::testMNIST(int train_record_cap, int test_record_cap)
{
	std::string train_image_fname = "C:\\Users\\charc\\source\\repos\\_SandboxProjects\\NeuralNet\\MNIST Database\\train-images.idx3-ubyte";
	std::string train_label_fname = "C:\\Users\\charc\\source\\repos\\_SandboxProjects\\NeuralNet\\MNIST Database\\train-labels.idx1-ubyte";
	std::string test_image_fname = "C:\\Users\\charc\\source\\repos\\_SandboxProjects\\NeuralNet\\MNIST Database\\t10k-images.idx3-ubyte";
	std::string test_label_fname = "C:\\Users\\charc\\source\\repos\\_SandboxProjects\\NeuralNet\\MNIST Database\\t10k-labels.idx1-ubyte";

	LabelParser train_labels(train_label_fname);
	ImageParser train_images(train_image_fname);
	LabelParser test_labels(test_label_fname);
	ImageParser test_images(test_image_fname);

	if (train_record_cap >= 0)
	{
		train_record_cap = train_record_cap < train_labels.getNumRecords() ? train_record_cap : train_labels.getNumRecords();
		train_labels.setRecordCap(train_record_cap);
		train_images.setRecordCap(train_record_cap);
	}
	if (train_record_cap >= 0)
	{
		test_record_cap = test_record_cap < test_labels.getNumRecords() ? test_record_cap : test_labels.getNumRecords();
		test_labels.setRecordCap(test_record_cap);
		test_images.setRecordCap(test_record_cap);
	}
	

	train_labels.parse();
	train_images.parse();
	test_labels.parse();
	test_images.parse();

	FFNeuralNet mnistNN(784, 16, 2, 10, 0.4);

	mnistNN.train(train_images.getData(), train_labels.getData(), 1.e-3);

	std::cout << "Performance = " << mnistNN.test(test_images.getData(), test_labels.getData()) * 100 << "%" << std::endl;

	return false;
}