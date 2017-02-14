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
#include "Graph.h"
#include "CommonFunctions.h"

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
		void initialize()
		{
			m_mode = -1;
			m_calibOutput.initialize();
		}
	}Input;

	typedef Data Output;

private:
	Input m_input;
	Output m_output;

	std::vector<Vertex*> m_vertices;

	FeatureExtractor m_feature;

	cv::VideoCapture cap[2];

	// camera param
	float baseLine;
	float covergence;
	float fX;
	float fY;
	float cX;
	float cY;
	float k1;
	float k2;
	float p1;
	float p2;
	cv::Matx33f K;
	cv::viz::Viz3d window1;
	// visualize with viz module
	cv::viz::Viz3d window;
	cv::viz::WCoordinateSystem wCoord;
	cv::viz::WGrid wGrid;
	
	std::vector<cv::Vec3b> m_color;
	// test full depth
	cv::Ptr<cv::StereoSGBM> sgbm;

	cv::Mat sumNew;
public:
	int m_mode = M_NULL;
	Stereo();
	~Stereo();

	inline void calculateDepth(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::Matx31f>& dst)
	{
		std::vector<float> disparity;
		calculateDepth(kp1, kp2, dst, disparity);
	}
	void calculateDepth(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::Matx31f>& dst, std::vector<float>& disparity);
	void run();
	void prevRun();
	void drawMap();
	bool save(char * dbPath);
	bool load(char * dbPath);
	void estimateRigid3D(std::vector<cv::Vec3f>& pt1, std::vector<cv::Vec3f>& pt2, cv::Matx33f& rot, cv::Matx31f& tran, double * error = nullptr);
	void estimateRigid3D(std::vector<cv::Point3f>& pt1, std::vector<cv::Point3f>& pt2, cv::Matx33f& rot, cv::Matx31f& tran, double* error = nullptr);
	//void setInput(Stereo::Input input);
	
	void keyframe();
	void saveImage(std::string fileName);
	cv::Mat loadImage(std::string fileName);

	void setInput(const Input input);
	void setInput(const Input input, const CalibOutput calibOutput);
	Output getOutput() const;
	void setCalibOutput(const CalibOutput output);
	void RnT2RT(cv::Matx33f & R, cv::Matx31f & T, cv::Matx44f & RT);
};

