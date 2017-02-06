#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <Windows.h>

#include "openCV.h"
#include "opencv2\ximgproc\disparity_filter.hpp"
#include "StereoCalibration.h"
#include "FeatureExtractor.h"
#include "Stereo.h"

#include "functions.h"

bool openCam(cv::VideoCapture* cap);
bool readCam(cv::VideoCapture* cap, cv::Mat* img);

typedef enum _mode_
{
	_NULL = -1, CALIBRATION = 0, UNDISTORTION = 1
}Mode;

int main(int argc, char** argv)
{
	int mode = _NULL;
	///
	std::ofstream log;
	log.open("data.txt");
	
	VideoCapture cap[2];
	cv::Mat img[2];
	if (!openCam(cap))
		return 0;

	//// Stereo Calibration
	StereoCalibration stereoCalib(7, 4, 0.026, 0.026);
	int nFrames = 50;		// the num of chessboard
	bool reg_chessboard = false;

	/// flag
	bool isCalibrated = false;

	//// FeatureExtractor
	FeatureExtractor fe;
	Stereo stereo;
	
	/*if (stereoCalib.LoadCalibrationData("..\\data\\camera\\")) {
		mode = UNDISTORTION;
		stereo.setCalibOutput(stereoCalib.getOutput());
	}*/

	IPC ipc("Stereo_App.exe");
	ipc.connect("Stereo_App.exe");
	ipc.start("Stereo_App.exe");

	Robot *robot = ipc.connect<Robot>("Robot_Pioneer.exe");
	if (robot == nullptr) {
		printf("Robot is not connected!\n");
		return 0;
	}

	robot->KeyFrame[0][0] = 0;
	robot->KeyFrame[0][1] = 1000;
	robot->KeyFrame[0][2] = 10;

	// main loop
	while (1)
	{
		Stereo::Input stInput;
		stInput.initialize();
		mode = _NULL;
		if (readCam(cap, img))
		{
			if (mode == UNDISTORTION)
			{
				stereoCalib.Undistort(img[0], stInput.m_leftImg, img[1], stInput.m_rightImg);
				//cv::waitKey(10);
			}
			else
			{
				stInput.m_leftImg = img[0];
				stInput.m_rightImg = img[1];
			}
			stereo.setCalibOutput(stereoCalib.getOutput());
			stereo.setInput(stInput);
		}
		else
			return 0;

		stereo.run();
		stereo.drawMap();

		if (GetAsyncKeyState(0x49) & 0x8000) {		// I
			printf("keyframed\n");
			stereo.keyframe();
		}
		if (robot->KeyFrame[0][0])
		{
			printf("keyframed\n");
			stereo.keyframe();
			robot->KeyFrame[0][0] = 1;
		}

		if (GetAsyncKeyState(0x53) & 0x8000) {		// S
			if (stereo.save("..\\data\\stereo.dat"))
				printf("ref point save done!\n");
			else
				printf("ref point save failed..\n");

			if(stereoCalib.isCalibed())
				if (stereoCalib.SaveCalibrationData("..\\data\\camera\\"))
					printf("calibration data save done!\n");
				else
					printf("calibration data save failed..\n");
		}

		if (GetAsyncKeyState(0x4C) & 0x8000) {		// L
			if (stereo.load("..\\data\\stereo.dat")) {
				printf("load done!\n");
				stereo.m_mode = Stereo::M_QUERY;
			}
			else
				printf("load failed..\n");
			if (stereoCalib.LoadCalibrationData("..\\data\\camera\\")) {
				printf("calibration data load done!\n");
				mode = UNDISTORTION;
			}
			else
				printf("calibration data load failed..\n");
		}
		if (GetAsyncKeyState(0x51) & 0x8000) {		// Q
			printf("break! \n");
			break;
		}
		if (GetAsyncKeyState(0x43) & 0x8000) {		// C
			mode = CALIBRATION;
			printf("Calibration..\n");
		}

		if(mode == CALIBRATION)
		{
			reg_chessboard = true;
			int iFrames = 0;
			while (iFrames < nFrames)
			{
				readCam(cap, img);
				iFrames = stereoCalib.FindChessboard(img[0].clone(), img[1].clone(), reg_chessboard);
				printf("iFrame : nFrame = %d : %d\n", iFrames, nFrames);
			}
			if (stereoCalib.RunCalibration()) {
				destroyWindow("find chess corner");
				printf("Calibration Done!\n");
				mode = UNDISTORTION;
			}
			reg_chessboard = false;
		}
//		cv::Mat tmpImg;
//		cap.read(tmpImg);
//		leftImg = tmpImg(cv::Rect(0, 0, tmpImg.cols / 2, tmpImg.rows));
//		rightImg = tmpImg(cv::Rect(tmpImg.cols / 2, 0, tmpImg.cols / 2, tmpImg.rows));
//		//
//		cv::Mat rgbCanvas = tmpImg.clone();
//
//		// Calibration setting
//		if (mode == Mode::CALIBRATION)
//		{
//			int iframes = calib.FindChessboard(leftImg, rightImg, reg_chessboard);
//			if (iframes >= nFrames)
//			{
//				if (calib.RunCalibration())
//					mode = Mode::UNDISTORTION;
//				else
//					mode = Mode::_NULL;
//			}
//			reg_chessboard = false;
//
//			putText(rgbCanvas, cv::format("Recognized chessboard = %d/%d", iframes, nFrames), cv::Point(10, 25), 1, 1, cv::Scalar(0, 0, 255));
//		}
//		else if (mode == Mode::UNDISTORTION)
//		{
//			// distortion
//			cv::Mat tmp1 = leftImg.clone();
//			cv::Mat tmp2 = rightImg.clone();
//			//calib.Undistort(tmp1, leftImg, tmp2, rightImg);
//
//			FeatureExtractor::Input feInput;
//			feInput.m_LeftImg = tmp1;
//			feInput.m_RightImg = tmp2;
//			fe.setInput(feInput);
//			
//			fe.run();
//
//			FeatureExtractor::Output feOutput = fe.getOutput();
//			//fe.setInputImage(tmp1, tmp2);
//
//			isCalibrated = true;
//
//			sgbm->compute(tmp1, tmp2, disparity16S);
//			sm->compute(tmp2, tmp1, img16Sr);
//
//			cv::Mat showDisparity;
//			disparity16S.convertTo(showDisparity, CV_8UC1, 255 / (numberOfDisparities*16.));
//			///printf("disparity16S: %s %d x %d\n", type2str(disparity16S.type()).c_str(), disparity16S.rows, disparity16S.cols);
//			cv::imshow("disparity", showDisparity);
//
//			wls_filter->setLambda(lambda);
//			wls_filter->setSigmaColor(sigma);
//			wls_filter->filter(disparity16S, tmp1, filteredDisparity, img16Sr);
//
//			cv::Mat showFilteredDisparity;
//			filteredDisparity.convertTo(showFilteredDisparity, CV_8U, 255 / (numberOfDisparities*16.));
//			///printf("filteredDisparity: %s %d x %d\n", type2str(filteredDisparity.type()).c_str(), filteredDisparity.rows, filteredDisparity.cols);
//			cv::imshow("Filtered Disparity", showFilteredDisparity);
//
//			cv::Mat xyz;
//			// output : 3-channel floating-point image of the same size as disparity
//			cv::reprojectImageTo3D(filteredDisparity / 5, xyz, Q, true);
//
//
//			///printf("xyz: %s %d x %d\n", type2str(xyz.type()).c_str(), xyz.rows, xyz.cols);
//			cv::Mat pointCloud = xyz.clone();
//			for (int y = 0; y < xyz.rows; y++) {
//				for (int x = 0; x < xyz.cols; x++)
//				{
//					cv::Vec3f &data = pointCloud.at<cv::Vec3f>(y, x);
//					if (data[2] >= 1000.f || abs(data[0]) > 1000.f || abs(data[1]) > 1000.f)
//					{
//						data = cv::Vec3f();
//					}
//					else
//						data = -data/100.f;		// scaling 
//				}
//			}
//
//			cv::Mat showXYZ;
//			xyz.convertTo(showXYZ, CV_8UC3, 255 / (numberOfDisparities*8.));
//			cv::imshow("XYZ", showXYZ);
//			
//			//// Point Feature
//			// compute
//			
//
//			// convert 3d coordinates
//			cv::Mat pointXYZ(xyz.size(), xyz.type());
//			pointXYZ.setTo(0);
//			for (int i = 0; i <feOutput.m_leftKp.size(); i++) {
//				cv::Point2f pt = feOutput.m_leftKp.at(i).pt;
//				pointXYZ.at<cv::Vec3f>(pt.y + 0.5f, pt.x + 0.5f) = pointCloud.at<cv::Vec3f>(pt.y + 0.5f, pt.x + 0.5f);
//			}
//
//	/*		cv::Mat depth(xyz.size(), CV_32FC1);
//			for(int y=0;y<xyz.rows;y++)
//				for (int x = 0; x < xyz.cols; x++) {
//					if (filteredDisparity.at<float>(y, x) == 0.0) {
//						depth.at<float>(y, x) = 0.0;
//					}
//					else
//						depth.at<float>(y, x) = FX * baseLine / filteredDisparity.at<float>(y, x);
//				}
//			depth.convertTo(depth, CV_8UC1, 255 / (numberOfDisparities*8.));
//			cv::imshow("depth", depth);
//*/
//			//Mat cloudMat = Mat(xyz.rows, xyz.cols, CV_32FC3);
//
//			//for (int row = 0; row < xyz.rows; row++) {
//			//	for (int col = 0; col < xyz.cols; col++) {
//			//		Vec3f data = vecXYZ.at<Vec3f>(row, col);// Vec3f(xyz.at<Vec3f>(row, col)[0], xyz.at<Vec3f>(row, col)[1], xyz.at<Vec3f>(row, col)[2]);
//			//		cloudMat.at<Vec3f>(row, col) = data;
//			//	}
//			//}
//			cv::Mat colorMap(pointCloud.size(), CV_8UC3);
//			colorMap.setTo(0);
//			for (int i = 0; i < feOutput.m_leftKp.size(); i++) {
//				cv::Point pt = cv::Point((int)(feOutput.m_leftKp.at(i).pt.x + 0.5f), (int)(feOutput.m_leftKp.at(i).pt.y + 0.5f));
//				colorMap.at<cv::Vec3b>(pt)[0] = feOutput.m_color.at(i).val[0];
//				colorMap.at<cv::Vec3b>(pt)[1] = feOutput.m_color.at(i).val[1];
//				colorMap.at<cv::Vec3b>(pt)[2] = feOutput.m_color.at(i).val[2];
//			}
//			cv::imshow("te", colorMap);
//			cv::viz::WCloud cw(pointCloud, tmp1);
//			cw.setRenderingProperty(cv::viz::POINT_SIZE, 2);
//			window.showWidget("Cloud Widget", cw);
//			cv::viz::WCloud ptCloud(pointXYZ, colorMap);
//			ptCloud.setRenderingProperty(cv::viz::POINT_SIZE, 4);
//			window.showWidget("Point Feature Cloud", ptCloud);
//			
//			window.spinOnce(30, true);
//			
//		}
//
//		if (0)
//			for (int i = 0; i < rgbCanvas.rows; i += 16)
//				cv::line(rgbCanvas, cv::Point(0, i), cv::Point(rgbCanvas.cols, i), cv::Scalar(0, 255, 0));
//		
//		cv::imshow("canvas", rgbCanvas);
//		cmd = cv::waitKey(10);
//		if (cmd == 'd')
//			mode = Mode::UNDISTORTION;
//		if (cmd == 'r')
//			window.setViewerPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d()));
//		if (cmd == 's') {
//			cv::waitKey();
//		}

	std::cout << std::endl;
	cv::waitKey(30);
	}
	return 0;
}

bool openCam(cv::VideoCapture* cap)
{
	cap[0].open(0);
	cap[1].open(1);

	if (!cap[0].isOpened())
		return false;

	return true;
}

bool readCam(cv::VideoCapture* cap, cv::Mat* img)
{
	if (!cap[0].isOpened())
		return false;
	else if (!cap[1].isOpened())
	{
		cv::Mat tmpImg;
		if (!cap[0].read(tmpImg))
			return false;
		img[0] = tmpImg(cv::Rect(0, 0, tmpImg.cols / 2, tmpImg.rows)).clone();
		img[1] = tmpImg(cv::Rect(tmpImg.cols / 2, 0, tmpImg.cols / 2, tmpImg.rows)).clone();
	}
	else
	{
		if (!cap[0].read(img[0]) || !cap[1].read(img[1]))
			return false;
	}
	return true;
}