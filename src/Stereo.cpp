#include "Stereo.h"

Stereo::Stereo()
{
	baseLine = 120.0;
	covergence = 0.00285;
	fX = 700.0;
	fY = 700.0;
	cX = 320.0;
	cY = 240.0;
	k1 = -0.15;
	k2 = 0.0;
	p1 = 0.0;
	p2 = 0.0;

	window = cv::viz::Viz3d("Coordinate Frame");
	window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
	window.setViewerPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0.0, 0.0, -5.0)));

	// test
	sgbm = cv::StereoSGBM::create(0, 16 * 6, 9);

	int sgbmWinSize = 3;
	int numberOfDisparities = 16 * 6;
	int cn = 3;

	sgbm->setPreFilterCap(63);
	sgbm->setBlockSize(sgbmWinSize);
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
}


Stereo::~Stereo()
{
}

bool Stereo::openCam()
{
	cap[0].open(0);
	cap[1].open(1);

	if(!cap[0].isOpened())
		return false;

	return true;
}

bool Stereo::readCam()
{
	if(!cap[0].isOpened())
		return false;
	else if (!cap[1].isOpened())
	{
		cv::Mat tmpImg;
		if (!cap[0].read(tmpImg))
			return false;
		m_input.m_leftImg = tmpImg(cv::Rect(0, 0, tmpImg.cols / 2, tmpImg.rows)).clone();
		m_input.m_rightImg = tmpImg(cv::Rect(tmpImg.cols / 2, 0, tmpImg.cols / 2, tmpImg.rows)).clone();
	}
	else
	{
		if (!cap[0].read(m_input.m_leftImg) || !cap[1].read(m_input.m_rightImg))
			return false;
	}
	return true;
}

void Stereo::run()
{
	if (!readCam()) {
		printf("cameras is not opened\n");
		return;
	}

	FeatureExtractor::Input feInput;
	feInput.m_LeftImg = m_input.m_leftImg;
	feInput.m_RightImg = m_input.m_rightImg;

	m_feature.setInput(feInput);
	m_feature.run();

	FeatureExtractor::Output feOutput = m_feature.getOutput();

	// caculate depth
	cv::Mat depth(m_input.m_leftImg.size(), CV_32FC3, cv::Scalar::all(0));
	int size = feOutput.m_leftKp.size();
	for (int i = 0; i < size; i++)
	{
		cv::Point pt = cv::Point((int)(feOutput.m_leftKp.at(i).pt.x + 0.5f), (int)(feOutput.m_leftKp.at(i).pt.y + 0.5f));
		
		depth.at<cv::Vec3f>(pt)[2] = fX * baseLine / (feOutput.m_leftKp.at(i).pt.x - feOutput.m_rightKp.at(i).pt.x);		// Z
		depth.at<cv::Vec3f>(pt)[0] = (feOutput.m_leftKp.at(i).pt.x - cX) * depth.at<cv::Vec3f>(pt)[2] / fX;					// X
		depth.at<cv::Vec3f>(pt)[1] = (feOutput.m_leftKp.at(i).pt.y - cY) * depth.at<cv::Vec3f>(pt)[2] / fY;					// Y

		//std::cout << depth.at<cv::Vec3f>(pt) / 1000.f << std::endl;
	}

	//
	cv::Mat disparity16S(m_input.m_leftImg.size(), CV_16S);
	sgbm->compute(m_input.m_leftImg, m_input.m_rightImg, disparity16S);
	cv::Mat xyz;
	cv::Matx44d Q = cv::Matx44d(	// http://answers.opencv.org/question/4379/from-3d-point-cloud-to-disparity-map/
		1.0, 0.0, 0.0, -cX,
		0.0, 1.0, 0.0, -cY,
		0.0, 0.0, 0.0, fX,
		0.0, 0.0, -1.0 / baseLine, 0/*(CX - CX) / baseLine*/
	);
	cv::reprojectImageTo3D(disparity16S, xyz, Q, true);

	cv::Mat pointXYZ(xyz.size(), CV_32FC3, cv::Scalar::all(0));
	for (int i = 0; i < size; i++)
	{
		cv::Point pt = cv::Point((int)(feOutput.m_leftKp.at(i).pt.x + 0.5f), (int)(feOutput.m_leftKp.at(i).pt.y + 0.5f));
		if (abs(xyz.at<cv::Vec3f>(pt).val[0]) > 10000.f || abs(xyz.at<cv::Vec3f>(pt).val[1]) > 10000.f || abs(xyz.at<cv::Vec3f>(pt).val[2]) >= 10000.f)
			pointXYZ.at<cv::Vec3f>(pt) = cv::Vec3f();
		else
			pointXYZ.at<cv::Vec3f>(pt) = - xyz.at<cv::Vec3f>(pt) / 50.f;
		//std::cout << pt << ", " << pointXYZ.at<cv::Vec3f>(pt) <<std::endl;
	}


	// 
	cv::Mat pointCloud = depth.clone();
	for(int y=0;y<depth.rows;y++)
		for (int x = 0; x < depth.cols; x++)
		{
			cv::Vec3f &data = pointCloud.at<cv::Vec3f>(y, x);
			if (abs(data[2]) >= 10000.f || abs(data[0]) > 10000.f || abs(data[1]) > 10000.f)
				data = cv::Vec3f();
			else {
				data = data / 1000.f;
			}
		}

	
	// color mapping
	cv::Mat colorMap(depth.size(), CV_8UC3, cv::Scalar::all(0));
	for (int i = 0; i < size; i++)
	{
		cv::Point pt = cv::Point((int)(feOutput.m_leftKp.at(i).pt.x + 0.5f), (int)(feOutput.m_leftKp.at(i).pt.y + 0.5f));
		colorMap.at<cv::Vec3b>(pt) = feOutput.m_color.at(i);
	}
	cv::viz::WCloud cw(pointCloud, /*colorMap*/cv::viz::Color::yellow());
	cv::viz::WCloud cw2(pointXYZ, cv::viz::Color::pink());
	cw.setRenderingProperty(cv::viz::POINT_SIZE, 2);
	cw2.setRenderingProperty(cv::viz::POINT_SIZE, 2);
	window.showWidget("Cloud Widget", cw);
	window.showWidget("Cloud Widget2", cw2);
	window.showWidget("CameraPosition Widget", cv::viz::WCameraPosition(cv::Matx33d(fX, 0, cX, 0, fY, cY, 0, 0, 1)));
	window.spinOnce(30, true);
	
}
