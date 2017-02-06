#pragma once

#include <iostream>
#include <fstream>
#include "openCV.h"

//// main class
class FeatureExtractor
{
public:
	typedef enum _feature_extractor_mode
	{
		FE_POINT = 0, FE_LINE = 1
	}Mode;

	typedef struct _feature_extractor_input
	{
		cv::Mat m_LeftImg;
		cv::Mat m_RightImg;
	}Input;

	typedef struct _feature_extractor_output
	{
		std::vector<cv::KeyPoint> m_leftKp;
		std::vector<cv::KeyPoint> m_rightKp;
		cv::Mat m_leftDescr;
		cv::Mat m_rightDescr;
		std::vector<int> m_mappingIdx1;
		std::vector<int> m_mappingIdx2;
		std::vector<cv::Vec3b> m_color;
	}Output;

private:
	// point feature
	cv::Ptr<cv::AKAZE> akaze;
	double akaze_thresh;
	// line feature
	//cv::Ptr<cv::ximgproc::FastLineDetector> fsd;

	// matcher
	cv::BFMatcher matcher;
	cv::Matx33d K;
	double nn_match_ratio;
	double ransac_thresh;

	// input-output init
	Input m_input;
	Output m_output;

	int m_mode;

	std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keyPoints);
	void lineCompute();
public:
	FeatureExtractor();
	~FeatureExtractor();
	
	inline void setInput(const Input input)
	{
		// shallow copy
		m_input.m_LeftImg = input.m_LeftImg;
		m_input.m_RightImg = input.m_RightImg;
	}

	inline Output getOutput() const
	{
		return m_output;
	}
	void pointFeatureExtracte(cv::Mat & src, std::vector<cv::KeyPoint>& kp, cv::Mat & dscr);
	bool featureMatching(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat & dscr1, cv::Mat & dscr2, double* matched_ratio = nullptr);
	void allCompute();
	void run();
	static std::vector<cv::Vec3b> colorMapping(int Size);
};