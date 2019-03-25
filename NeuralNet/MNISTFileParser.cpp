#include "MNISTFileParser.h"

MNISTFileParser::MNISTFileParser(std::string filename)
	: filename(filename)
{
	data = NULL;
	fs.open(filename, std::ios::binary);
	std::cout << "Opening file:\n\t" << filename << "\n\t";
	if (fs.good())
	{
		unsigned char magnum[8];
		for (int i = 0; i < 8; i++)
			fs >> magnum[i];
		magic_number = magnum[0] * 65536 + magnum[1] * 4096 + magnum[2] * 256 + magnum[3];
		num_records = magnum[4] * 65536 + magnum[5] * 4096 + magnum[6] * 256 + magnum[7];
		record_cap = -1;
		std::cout << "magic number = " << magic_number << ", number of records = " << num_records << std::endl << std::endl;
	}
	else
		throw std::runtime_error("Error opening data file");
}



LabelParser::LabelParser(std::string filename)
	: MNISTFileParser(filename)
{
	if (magic_number != 2049)
		throw std::runtime_error("Magic number error: file is not a Label Data file");
	data = new Eigen::MatrixXd();
	
}

void LabelParser::parse()
{
	std::cout << "Parsing " << filename << std::endl;
	std::cout << "0%|         |         |100%\n   ";

	if (record_cap >= 0)
		num_records = num_records < record_cap ? num_records : record_cap;

	(*data) = Eigen::MatrixXd::Zero(10, num_records);

	unsigned char byte;
	for (int i = 0; i < num_records; i++)
	{
		fs >> byte;
		(*data)(byte, i) = 1.0;

		if (!(i % (num_records / 20)))
		{
			std::cout << ".";
		}
	}
	std::cout << std::endl;
}



ImageParser::ImageParser(std::string filename)
	: MNISTFileParser(filename)
{
	if (magic_number != 2051)
		throw std::runtime_error("Magic number error: file is not a Image Data file");
	unsigned char magnum[8];
	for (int i = 0; i < 8; i++)
		fs >> magnum[i];
	image_rows = magnum[0] * 65536 + magnum[1] * 4096 + magnum[2] * 256 + magnum[3];
	image_cols = magnum[4] * 65536 + magnum[5] * 4096 + magnum[6] * 256 + magnum[7];

	data = new Eigen::MatrixXd();
	

}

void ImageParser::parse()
{
	std::cout << "Parsing " << filename << std::endl;
	std::cout << "0\%|         |         |100\%\n   ";

	if (record_cap >= 0)
		num_records = num_records < record_cap ? num_records : record_cap;

	(*data) = Eigen::MatrixXd::Zero(image_rows*image_cols, num_records);

	unsigned char byte;
	for (int i = 0; i < num_records; i++)
	{
		for (int j = 0; j < image_rows*image_cols; j++)
		{
			fs >> byte;
			(*data)(j, i) = (double)byte / 255.0;
		}

		if (!(i % (num_records / 20)))
		{
			std::cout << ".";
		}
	}
	std::cout << std::endl;
}