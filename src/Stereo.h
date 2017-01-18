#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <tuple>
#include "openCV.h"
#include "opencv2\ximgproc\disparity_filter.hpp"
#include "StereoCalibration.h"
#include "FeatureExtractor.h"

typedef StereoCalibration::Output CalibOutput;

class Stereo
{
public:
	typedef enum _stereo_mode
	{
		M_NULL = -1, M_DB = 0, M_QUERY = 1
	}Mode;
	typedef struct _stereo_input
	{
		cv::Mat m_leftImg;
		cv::Mat m_rightImg;
		int m_mode;
		CalibOutput m_calibOutput;
	}Input;

	typedef struct _stereo_database
	{
		std::vector<cv::KeyPoint> m_vecKeyPoint;		// keypoint
		cv::Mat m_vecDescriptor;						// Descriptor
		std::vector<cv::Vec3f> m_vecWorldCoord;			// 3D coordinates
		cv::Mat R, T;									// Rotation, Translation Matrix
	}DB;

	typedef struct _stereo_output
	{
		std::vector<cv::KeyPoint> m_KeyPoint;
		cv::Mat m_Descriptor;
		std::vector<cv::Vec3f> m_WorldCoord;
		cv::Mat R, T;
	}Output;
private:
	Input m_input;
	Output m_output;
	std::vector<DB> m_vecDB;

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
	cv::Matx33d K;
	cv::viz::Viz3d window1;
	// visualize with viz module
	cv::viz::Viz3d window;

	// test full depth
	cv::Ptr<cv::StereoSGBM> sgbm;
	
public:
	int m_mode = M_NULL;
	Stereo();
	~Stereo();

	inline void caculateDepth(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::Vec3f>& dst)
	{
		std::vector<float> disparity;
		caculateDepth(kp1, kp2, dst, disparity);
	}
	void caculateDepth(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::Vec3f>& dst, std::vector<float>& disparity);
	void run();
	void prevRun();
	bool save(char * dbPath);
	bool load(char * dbPath);
	void estimateRigid3D(std::vector<cv::Vec3f>& pt1, std::vector<cv::Vec3f>& pt2, cv::Matx<double, 3, 3>& rot, cv::Matx<double, 3, 1>& tran, double * error = nullptr);
	void estimateRigid3D(std::vector<cv::Point3f>& pt1, std::vector<cv::Point3f>& pt2, cv::Matx<double, 3, 3>& rot, cv::Matx<double, 3, 1>& tran, double* error = nullptr);
	//void setInput(Stereo::Input input);
	
	void keyframe();
	void saveImage(std::string fileName);
	cv::Mat loadImage(std::string fileName);

	void setInput(const Input input);
	void setInput(const Input input, const CalibOutput calibOutput);
	Output getOutput() const;
	void setCalibOutput(const CalibOutput output);
};

