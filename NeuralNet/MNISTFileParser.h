#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>


class MNISTFileParser
{
public:
	MNISTFileParser(std::string filename);
	~MNISTFileParser() 
	{
		fs.close();
		if(data != NULL) delete data;
	}

	virtual void parse(void) = 0;

	Eigen::MatrixXd* getData(void) const { return data; }

	std::pair<int, int> dataSize(void) const { return std::pair<int, int>((*data).rows(), (*data).cols()); };

	void setRecordCap(int cap) { record_cap = cap; };

	int getNumRecords(void) const { return num_records; };
protected:
	const std::string filename;
	int magic_number;
	int num_records;
	Eigen::MatrixXd* data;
	std::ifstream fs;
	int record_cap;
};


class LabelParser : public MNISTFileParser
{
public:
	LabelParser(std::string filename);
	virtual void parse();
};


class ImageParser : public MNISTFileParser
{
public:
	ImageParser(std::string filename);
	virtual void parse();

private:
	int image_rows;
	int image_cols;

};