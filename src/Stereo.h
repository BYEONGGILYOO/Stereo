#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "openCV.h"
#include "opencv2\ximgproc\disparity_filter.hpp"
#include "StereoCalibration.h"
#include "FeatureExtractor.h"

class Stereo
{
public:
	typedef struct _stereo_input
	{
		cv::Mat m_leftImg;
		cv::Mat m_rightImg;
	}Input;

	typedef struct _stereo_output
	{

	}Output;
private:
	Input m_input;
	Output m_output;

	FeatureExtractor m_feature;

	cv::VideoCapture cap[2];

	// camera param
	double baseLine;
	double covergence;
	double fX;
	double fY;
	double cX;
	double cY;
	double k1;
	double k2;
	double p1;
	double p2;

	// visualize with viz module
	cv::viz::Viz3d window;

	// test full depth
	cv::Ptr<cv::StereoSGBM> sgbm;
	
public:
	Stereo();
	~Stereo();
	bool openCam();
	bool readCam();
	void run();
};

