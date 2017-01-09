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
std::string type2str(int type);

typedef enum _mode_
{
	_NULL = -1, CALIBRATION = 0, UNDISTORTION = 1
}Mode;

int main(int argc, char** argv)
{
	//cv::VideoCapture cap;
	//cap.open(0);
	//if (!cap.isOpened())
	//{
	//	std::cout << "Capture could not be opened successfully" << std::endl;
	//	system("pause");
	//	return 0;
	//}
	//// rgb data
	//cv::Mat leftImg, rightImg;
	//// d
	//cv::Mat disparity16S = cv::Mat(cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH) / 2, CV_16S);
	//cv::Mat img16Sr = cv::Mat(cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH) / 2, CV_16S);

	//cv::Mat filteredDisparity;

	//int mode = -1;
	//int cmd = 0;

	//// chess board info.
	//int board_w = 4;
	//int board_h = 7;
	//float square_w = 0.314f / 9;
	//float square_h = 0.209f / 6;
	//bool lineOn = true;
	//bool reg_chessboard = false;
	//const int nFrames = 30;
	//StereoCalibration calib(board_w, board_h, square_w, square_h);

	//// calibration param
	//double baseLine = 120.0;
	//double covergence = 0.00285;
	//double FX = 700.0;
	//double FY = 700.0;
	//double CX = 320.0;
	//double CY = 240.0;
	//double K1 = -0.15;
	//double K2 = 0.0;
	//double P1 = 0.0;
	//double P2 = 0.0;

	//cv::Matx33d K = cv::Matx33d(FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0);
	//cv::Matx41d distCoeffs = cv::Matx41d(K1, K2, P1, P2);
	//cv::Matx44d Q = cv::Matx44d(	// http://answers.opencv.org/question/4379/from-3d-point-cloud-to-disparity-map/
	//	1.0, 0.0, 0.0, -CX,
	//	0.0, 1.0, 0.0, -CY,
	//	0.0, 0.0, 0.0, FX,
	//	0.0, 0.0, -1.0 / baseLine, 0/*(CX - CX) / baseLine*/
	//);
	//cv::Mat Q_32F = cv::Mat::eye(4, 4, CV_32FC1);
	//Q_32F.at<float>(0, 3) = -CX;
	//Q_32F.at<float>(1, 3) = -CY;
	//Q_32F.at<float>(2, 2) = 0;
	//Q_32F.at<float>(2, 3) = FX;	
	//Q_32F.at<float>(3, 2) = -1.0 / baseLine;
	//Q_32F.at<float>(3, 3) = 0;

	////// SGBM
	//cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16 * 6, 9);
	////Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,    //int minDisparity
	////	96,     //int numDisparities
	////	5,      //int SADWindowSize
	////	600,    //int P1 = 0
	////	2400,   //int P2 = 0
	////	20,     //int disp12MaxDiff = 0
	////	16,     //int preFilterCap = 0
	////	1,      //int uniquenessRatio = 0
	////	100,    //int speckleWindowSize = 0
	////	20,     //int speckleRange = 0
	////	true);  //bool fullDP = false

	//// param
	//int sgbmWinSize = 3;
	//int numberOfDisparities = 16 * 6;
	//int cn = 3;

	//// init
	//sgbm->setPreFilterCap(63);
	//sgbm->setBlockSize(sgbmWinSize);
	//sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	//sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	//sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);


	//// filter
	//cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
	//wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
	//cv::Ptr<cv::StereoMatcher> sm = cv::ximgproc::createRightMatcher(sgbm);
	//// param
	//double lambda = 8000.0;
	//double sigma = 1.5;
	//double vis_multi = 1.0;

	////// viz
	//cv::viz::Viz3d window("Coordinate Frame1");
	//window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
	//window.setViewerPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d()));

	///
	std::ofstream log;
	log.open("data.txt");
	
	/// flag
	bool isCalibrated = false;

	//// FeatureExtractor
	FeatureExtractor fe;
	Stereo stereo;
	if (!stereo.openCam())
		return 0;
	// main loop
	while (1)
	{
		stereo.run();

		if (GetAsyncKeyState(0x49) & 0x8000) {		// I
			printf("keyframed\n");
			stereo.keyframe();
		}
		if (GetAsyncKeyState(0x53) & 0x8000) {		// S
			if (stereo.save("..\\db\\stereo.dat"))
				printf("save done!\n");
			else
				printf("save failed..\n");
		}
		if (GetAsyncKeyState(0x4C) & 0x8000) {		// L
			if (stereo.load("..\\db\\stereo.dat"))
				printf("load done!\n");
			else
				printf("load failed..\n");
		}
		if (GetAsyncKeyState(0x51) & 0x8000) {		// Q
			printf("break! \n");
			break;
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
	}
	return 0;
}

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//
///**
//* @file transformations.cpp
//* @brief Visualizing cloud in different positions, coordinate frames, camera frustums
//* @author Ozan Cagri Tonkal
//*/
//
//using namespace cv;
//using namespace std;
//
///**
//* @function help
//* @brief Display instructions to use this tutorial program
//*/
//void help()
//{
//	cout
//		<< "--------------------------------------------------------------------------" << endl
//		<< "This program shows how to use makeTransformToGlobal() to compute required pose,"
//		<< "how to use makeCameraPose and Viz3d::setViewerPose. You can observe the scene "
//		<< "from camera point of view (C) or global point of view (G)" << endl
//		<< "Usage:" << endl
//		<< "./transformations [ G | C ]" << endl
//		<< endl;
//}

///**
//* @function cvcloud_load
//* @brief load bunny.ply
//*/
//Mat cvcloud_load()
//{
//	Mat cloud(1, 1889, CV_32FC3);
//	ifstream ifs("bunny.ply");
//
//	string str;
//	for (size_t i = 0; i < 12; ++i)
//		getline(ifs, str);
//
//	Point3f* data = cloud.ptr<cv::Point3f>();
//	float dummy1, dummy2;
//	for (size_t i = 0; i < 1889; ++i)
//		ifs >> data[i].x >> data[i].y >> data[i].z >> dummy1 >> dummy2;
//
//	cloud *= 5.0f;
//	return cloud;
//}
//
///**
//* @function main
//*/
//int main(int argn, char **argv)
//{
//
//	bool camera_pov = false;
//
//	/// Create a window
//	viz::Viz3d myWindow("Coordinate Frame");
//
//	/// Add coordinate axes
//	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
//
//	/// Let's assume camera has the following properties
//	Vec3f cam_pos(3.0f, 3.0f, 3.0f), cam_focal_point(3.0f, 3.0f, 2.0f), cam_y_dir(-1.0f, 0.0f, 0.0f);
//
//	/// We can get the pose of the cam using makeCameraPose
//	Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
//
//	/// We can get the transformation matrix from camera coordinate system to global using
//	/// - makeTransformToGlobal. We need the axes of the camera
//	Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f, -1.0f, 0.0f), Vec3f(-1.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), cam_pos);
//
//	/// Create a cloud widget.
//	Mat bunny_cloud = cvcloud_load();
//	
//	for (int row = 0; row < bunny_cloud.rows; row++) {
//		for (int col = 0; col < bunny_cloud.cols; col++) {
//			Vec3f data = Vec3f(bunny_cloud.at<Vec3f>(row, col)[0], bunny_cloud.at<Vec3f>(row, col)[1], bunny_cloud.at<Vec3f>(row, col)[2]);
//			bunny_cloud.at<Vec3f>(row, col) = data;
//		}
//	}
//	ofstream data;
//	data.open("data.txt");
//	for (int y = 0; y < bunny_cloud.rows; y++) {
//		for (int x = 0; x < bunny_cloud.cols * 3; x++) {
//			data << bunny_cloud.at<float>(y, x) << "\t";
//		}
//		data << "\n\n";
//	}
//	data.close();
//
//	viz::WCloud cloud_widget(bunny_cloud, viz::Color::green());
//
//	/// Pose of the widget in camera frame
//	Affine3f cloud_pose = Affine3f().translate(Vec3f(0.0f, 0.0f, 3.0f));
//	/// Pose of the widget in global frame
//	Affine3f cloud_pose_global = transform * cloud_pose;
//
//	/// Visualize camera frame
//	if (!camera_pov)
//	{
//		viz::WCameraPosition cpw(0.5); // Coordinate axes
//		viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
//		myWindow.showWidget("CPW", cpw, cam_pose);
//		myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
//	}
//
//	/// Visualize widget
//	myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);
//
//	/// Set the viewer pose to that of camera
//	if (camera_pov)
//		myWindow.setViewerPose(cam_pose);
//
//	/// Start event loop.
//	myWindow.spin();
//
//	return 0;
//}
