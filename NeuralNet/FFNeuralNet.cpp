#include "FFNeuralNet.h"


FFNeuralNet::FFNeuralNet(
	int nInputNodes, int nHiddenNodes, int nHiddenLayers, int nOutputNodes,
	double rate,
	std::pair<std::function<Eigen::MatrixXd(Eigen::MatrixXd)>,
	std::function<Eigen::MatrixXd(Eigen::MatrixXd)>> ACTFUN)
	: 
	nInputNodes(nInputNodes), nHiddenNodes(nHiddenNodes), nHiddenLayers(nHiddenLayers), nOutputNodes(nOutputNodes),
	rate(rate),
	actFun(ACTFUN.first), actFun_prime(ACTFUN.second),
	netDepth(nHiddenLayers+1)
{
	if (nHiddenNodes == 0 || nHiddenLayers == 0)
		throw std::runtime_error("FFNeuralNet must have potive number of hidden nodes/layers");

	activations.push_back(Eigen::MatrixXd::Random(nInputNodes, 1));
	grad_activations.push_back(Eigen::MatrixXd::Random(nInputNodes, 1));

	// Initialize input layer parameters:
	weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nInputNodes));
	prev_weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nInputNodes));
	grad_weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nInputNodes));
	prev_grad_weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nInputNodes));

	biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes,1));
	prev_biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));
	grad_biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));
	prev_grad_biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));

	activations.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1).cwiseAbs());
	grad_activations.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1).cwiseAbs());

	// Initialize hidden layer parameters:
	for (int i = 0; i < nHiddenLayers - 1; ++i)
	{
		weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nHiddenNodes));
		prev_weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nHiddenNodes));
		grad_weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nHiddenNodes));
		prev_grad_weights.push_back(Eigen::MatrixXd::Random(nHiddenNodes, nHiddenNodes));

		biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));
		prev_biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));
		grad_biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));
		prev_grad_biases.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1));

		activations.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1).cwiseAbs());
		grad_activations.push_back(Eigen::MatrixXd::Random(nHiddenNodes, 1).cwiseAbs());
	}

	// Initialized output layer parameters:
	weights.push_back(Eigen::MatrixXd::Random(nOutputNodes, nHiddenNodes));
	prev_weights.push_back(Eigen::MatrixXd::Random(nOutputNodes, nHiddenNodes));
	grad_weights.push_back(Eigen::MatrixXd::Random(nOutputNodes, nHiddenNodes));
	prev_grad_weights.push_back(Eigen::MatrixXd::Random(nOutputNodes, nHiddenNodes));

	biases.push_back(Eigen::MatrixXd::Random(nOutputNodes, 1));
	prev_biases.push_back(Eigen::MatrixXd::Random(nOutputNodes, 1));
	grad_biases.push_back(Eigen::MatrixXd::Random(nOutputNodes, 1));
	prev_grad_biases.push_back(Eigen::MatrixXd::Random(nOutputNodes, 1));

	activations.push_back(Eigen::MatrixXd::Random(nOutputNodes, 1).cwiseAbs());
	grad_activations.push_back(Eigen::MatrixXd::Random(nOutputNodes, 1).cwiseAbs());

}

Eigen::MatrixXd
FFNeuralNet::predict(Eigen::MatrixXd input)
{
	// todo validate input dimensions with nInputNodes
#ifdef DEBUG
	std::cout << "In predict():" << std::endl;
	
#endif
	activations[0] = input;
	Eigen::MatrixXd z_i;
	for (int i = 0; i < netDepth; i++)
	{
		z_i = weights[i] * activations[i] + biases[i];

#if 0
		std::cout << "Layer: " << i << std::endl;
		std::cout << "weights[backPropLayer] = " << weights[i].transpose() << std::endl;
		std::cout << "biases[backPropLayer] = " << biases[i].transpose() << std::endl;
		std::cout << "z_L = w*a+b = \n" << z_i.transpose() << std::endl;
#endif

		activations[i + 1] = actFun(z_i);

#if 0
		std::cout << activations[i].transpose() << "\n  -->\n\t  " << activations[i + 1].transpose() << std::endl;
		std::cin.get();
#endif

	}
 
	return *(activations.end()-1);
}

void //for now. Possibly return some kind of fit metrics?
FFNeuralNet::train(Eigen::MatrixXd* X, Eigen::MatrixXd* Y, double tol)
{

	double err = tol + 1.;
	double err_prev;
	size_t nItrs = 0;

	// Use 0.1% of data for stochastic gradient descent
	int sgd_samplsz = 
		10
		//(*X).cols() / 1000
		;
	std::vector<int> sgd_samples(sgd_samplsz, 0);
	do
	{
		// Randomly sample indices for stochastic gradient descent:
		for (int i = 0; i < sgd_samplsz; i++)
			sgd_samples[i] = rand() % (*X).cols();

		for (int i = 0; i < sgd_samplsz; i++)
		{

			//std::cout << grad_biases[0].transpose() << std::endl;
			//std::cout << grad_biases[1].transpose() << std::endl;
			//std::cout << grad_biases[2].transpose() << std::endl;

			prev_grad_biases = grad_biases;
			prev_grad_weights = grad_weights;

			err = backPropGradient((*X).col(i), (*Y).col(i));

			//std::cout << grad_biases[0].transpose() << std::endl;
			//std::cout << grad_biases[1].transpose() << std::endl;
			//std::cout << grad_biases[2].transpose() << std::endl;

			for (int i = 0; i < netDepth; i++)
			{
				Eigen::MatrixXd delta_bias = grad_biases[i], delta_weight = grad_weights[i];

				double gamma_numer =
					((biases[i] - prev_biases[i]).transpose() * (grad_biases[i] - prev_grad_biases[i])).trace();
				double gamma_denom =
					((grad_biases[i] - prev_grad_biases[i]).transpose() * (grad_biases[i] - prev_grad_biases[i])).trace() + 1;
				double gamma = gamma_numer / gamma_denom;

				//double gamma;
				double alpha = 0.1;
				gamma = rate;

				delta_bias = grad_biases[i].array() * (gamma * (1.0 - alpha) * -1.0) + prev_grad_biases[i].array() * alpha;

#if 0
				std::cout << "|bias_i - prev_bias_i|^2 = " << ((biases[i] - prev_biases[i]).transpose() * (biases[i] - prev_biases[i])).trace() << std::endl;
				std::cout << "bias_rate: gamma =\t" << gamma << ",\tgamma_numer = " << gamma_numer << "\tgamma_denom = " << gamma_denom << std::endl;
				std::cout << "|delta_bias|^2 = " << (delta_bias.transpose() * delta_bias).trace() << std::endl;
#endif

				prev_biases[i] = biases[i];
				biases[i] += delta_bias;

				gamma_numer =
					((weights[i] - prev_weights[i]).transpose() * (grad_weights[i] - prev_grad_weights[i])).trace();
				gamma_denom =
					((grad_weights[i] - prev_grad_weights[i]).transpose() * (grad_weights[i] - prev_grad_weights[i])).trace() + 1;

				gamma = gamma_numer / gamma_denom;

				//gamma = 0.3;

				delta_weight = grad_weights[i].array() * (gamma * (1.0 - alpha) * -1.0) + prev_grad_weights[i].array() * alpha;

#if 0
				std::cout << "|weight_i - prev_weight_i|^2 = " << ((weights[i] - prev_weights[i]).transpose() * (weights[i] - prev_weights[i])).trace() << std::endl;
				std::cout << "weight_rate: gamma =\t" << gamma << ",\tgamma_numer = " << gamma_numer << "\tgamma_denom = " << gamma_denom << std::endl;
				std::cout << "|delta_weight|^2 = " << (delta_weight.transpose() * delta_weight).trace() << std::endl << std::endl;
#endif

				prev_weights[i] = weights[i];
				weights[i] += delta_weight;
			}
		}
		std::cout << "Iteration: \t" << ++nItrs << "\t ||grad|| = \t " << err << std::endl;

#if 0
		std::cin.get();
#endif

	} while (err > tol);
	std::cout << "\nTotal Iterations = " << nItrs << std::endl << std::endl;
}

double
FFNeuralNet::test(Eigen::MatrixXd* X, Eigen::MatrixXd* Y)
{
	int pass = 0;
	Eigen::MatrixXd result;

	for (int i = 0; i < (*X).cols(); i++)
	{
		result = predict((*X).col(i));
		double resmax = -1.e99;
		int resmaxind = -1;
		for (int i = 0; i < result.rows(); i++)
		{
			if (result(i, 0) > resmax)
			{
				resmax = result(i, 0);
				resmaxind = i;
			}
		}
		if ( (*Y).col(i)(resmaxind,0) )
			pass++;
	}
	return (double)pass / (double)(*X).cols();
}

double
FFNeuralNet::backPropGradient(const Eigen::MatrixXd Xi, const Eigen::MatrixXd Yi)
{
/*

Directly compute the output layer gradient wrt output layer weights and biases

Directly compute dE/da_j for nodes a_j in (output layer - 1)

For step l = (ouput layer - 1) to l = (input layer):
	Using dE/da_j for a_j in the j+1-th layer,
	Directly compute layer l's gradient wrt output layer's l weights/biases
	Directly compute dE/da_j for a_j in the j-th (current) layer

At each step we aggregate into a gradient vector all the computed dE/d(weights,biases at each layer)
We then use this gradient in the train() method.

Would this function be most useful (for possibly other applications) if we have the gradient as a return value rather than updating a data member?
We could maybe make this a public function, and take an I/O pair as parameters and return the resulting back-propogation-calculated gradient?

*/
	int backPropLayer = netDepth;
	double grad_norm = 0.0;

	// Last activation layer is output layer -- this gradient is just the difference between output/train data:
	grad_activations[backPropLayer] = predict(Xi) - Yi;

#ifdef DEBUG
	std::cout << "Xi = " << Xi.transpose() << std::endl << "Yi = " << Yi.transpose() << std::endl;
	std::cout << "Prediction = " << (*(activations.end() - 1)).transpose() << std::endl;
	std::cout << "Error = " << (grad_activations[backPropLayer].transpose()*grad_activations[backPropLayer]).trace() * 0.5 << std::endl;
#endif

	// Now back-propogate up to input layer:
	while (--backPropLayer > -1)
	{
		//					  size = netDepth;        size = netDepth + 1;        size = netDepth;
		Eigen::MatrixXd z_L = weights[backPropLayer] * activations[backPropLayer] + biases[backPropLayer];

#ifdef DEBUG
		std::cout << "Layer: " << backPropLayer << std::endl;
		std::cout << "weights[backPropLayer] = \n" << weights[backPropLayer].transpose() << std::endl;
		std::cout << "biases[backPropLayer] = " << biases[backPropLayer].transpose() << std::endl;
		std::cout << "z_L = w*a+b = " << z_L.transpose() << std::endl;
		std::cout << "actFun(z_L) = " << activations[backPropLayer+1].transpose() << std::endl;
#endif

		// Derivative of C wrt biases -- this is daj/dzj*dC/daj term:
		grad_biases[backPropLayer] = actFun_prime(z_L).cwiseProduct(grad_activations[backPropLayer+1]);
		grad_norm += (grad_biases[backPropLayer].transpose()*grad_biases[backPropLayer]).trace();

		// Derivative of C wrt weights -- this is dzj/dwjk*daj/dzj*dC/daj term:
		grad_weights[backPropLayer] = grad_biases[backPropLayer] * activations[backPropLayer].transpose();
		grad_norm += (grad_weights[backPropLayer].transpose()*grad_weights[backPropLayer]).trace();

		// Derivative of C wrt prev layer activations -- this is Sum_j(dzj/dak(L-1)*daj(L)/dzj*dC/daj) term:
		grad_activations[backPropLayer] = weights[backPropLayer].transpose() * grad_biases[backPropLayer];

#ifdef DEBUG
		std::cout << "grad_biases[backPropLayer] = " << grad_biases[backPropLayer].transpose() << std::endl;
		std::cout << "grad_weights[backPropLayer] = \n" << grad_weights[backPropLayer].transpose() << std::endl;
		std::cin.get();
#endif


	}

	// Return norm of gradient of cost wrt weights/biases
	return pow(grad_norm,0.5);
}