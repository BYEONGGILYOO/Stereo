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
		cv::Vec3f &threeD = depth.at<cv::Vec3f>(pt);
		threeD.val[2] = fX * baseLine / (feOutput.m_leftKp.at(i).pt.x - feOutput.m_rightKp.at(i).pt.x);		// Z
		threeD.val[0] = (feOutput.m_leftKp.at(i).pt.x - cX) * threeD.val[2] / fX;							// X
		threeD.val[1] = (feOutput.m_leftKp.at(i).pt.y - cY) * threeD.val[2] / fY;							// Y
		
		//std::cout << depth.at<cv::Vec3f>(pt) / 1000.f << std::endl;
		m_output.m_WorldCoord.push_back(threeD);
	}
	m_output.m_Descriptor = feOutput.m_leftDescr;
	m_output.m_KeyPoint = feOutput.m_leftKp;
	// 
	cv::Mat pointCloud = depth.clone();
	for (int y = 0; y<depth.rows; y++)
		for (int x = 0; x < depth.cols; x++)
		{
			cv::Vec3f &data = pointCloud.at<cv::Vec3f>(y, x);
			if (abs(data[2]) >= 10000.f || abs(data[0]) > 10000.f || abs(data[1]) > 10000.f)
				data = cv::Vec3f();
			else {
				data = data / 1000.f;
			}
		}

	//
	if (0)
	{
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
				pointXYZ.at<cv::Vec3f>(pt) = -xyz.at<cv::Vec3f>(pt) / 50.f;
			//std::cout << pt << ", " << pointXYZ.at<cv::Vec3f>(pt) <<std::endl;
		}

		cv::viz::WCloud cw2(pointXYZ, cv::viz::Color::pink());
		cw2.setRenderingProperty(cv::viz::POINT_SIZE, 2);
		window.showWidget("Cloud Widget2", cw2);
	}
	
	// color mapping
	cv::Mat colorMap(depth.size(), CV_8UC3, cv::Scalar::all(0));
	for (int i = 0; i < size; i++)
	{
		cv::Point pt = cv::Point((int)(feOutput.m_leftKp.at(i).pt.x + 0.5f), (int)(feOutput.m_leftKp.at(i).pt.y + 0.5f));
		colorMap.at<cv::Vec3b>(pt) = feOutput.m_color.at(i);
	}
	cv::viz::WCloud cw(pointCloud, colorMap/*cv::viz::Color::yellow()*/);
	cw.setRenderingProperty(cv::viz::POINT_SIZE, 2);
	window.showWidget("Cloud Widget", cw);
	window.showWidget("CameraPosition Widget", cv::viz::WCameraPosition(cv::Matx33d(fX, 0, cX, 0, fY, cY, 0, 0, 1)));

	///////////////
	if (m_vecDB.size() > 4)
	{

	}
	
	
	window.spinOnce(30, true);
}

bool Stereo::save(char* dbPath)
{
	std::ofstream DBsaver(dbPath, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
	if (!DBsaver.is_open())
	{
		return false;
	}
	//// Keyframe[size, data], 3D coord, Descriptor[size, data]
	int nDocSize = m_vecDB.size();

	//char flag;
	//printf("Documet size: %d\n[Yes: Y, No: N]", nDocSize);
	//scanf(&flag);
	//if (flag != 'Y' && flag != 'y')
	//	return false;

	DBsaver.write((char*)&nDocSize, sizeof(int));

	for (int i = 0; i < nDocSize; i++)
	{
		// keypoint
		int nKP = (int)m_vecDB.at(i).m_vecKeyPoint.size();
		DBsaver.write((char*)&nKP, sizeof(int));

		for (int j = 0; j < nKP; j++) {
			cv::KeyPoint &kp = m_vecDB.at(i).m_vecKeyPoint.at(j);
			DBsaver.write((char*)&kp.angle, sizeof(float));
			DBsaver.write((char*)&kp.class_id, sizeof(int));
			DBsaver.write((char*)&kp.octave, sizeof(int));
			DBsaver.write((char*)&kp.pt.x, sizeof(float));
			DBsaver.write((char*)&kp.pt.y, sizeof(float));
			DBsaver.write((char*)&kp.response, sizeof(float));
			DBsaver.write((char*)&kp.size, sizeof(float));

			// 3D coordinates
			cv::Vec3f &threeD = m_vecDB.at(i).m_vecWorldCoord.at(j);
			DBsaver.write((char*)&threeD.val[0], sizeof(float));
			DBsaver.write((char*)&threeD.val[1], sizeof(float));
			DBsaver.write((char*)&threeD.val[2], sizeof(float));
		}

		// descriptor
		cv::Mat &dscr = m_vecDB.at(i).m_vecDescriptor;
		int dscr_type = dscr.type();
		int dscr_nData = 0;
		if (dscr_type == 0) dscr_nData = 1;
		else if (dscr_type == 5) dscr_nData = 4;
		else if (dscr_type == 6) dscr_nData = 8;
		else return false;

		DBsaver.write((char*)&dscr.cols, sizeof(int));
		DBsaver.write((char*)&dscr.rows, sizeof(int));
		DBsaver.write((char*)&dscr_type, sizeof(int));
		DBsaver.write((char*)dscr.data, dscr.cols * dscr.rows * dscr_nData);
	}

	DBsaver.close();
	return true;
}

bool Stereo::load(char * dbPath)
{
	m_vecDB.clear();

	std::ifstream DBloader(dbPath, std::ios_base::in | std::ios_base::binary);
	if(!DBloader.is_open())
		return false;

	int nDocSize = 0;
	DBloader.read((char*)&nDocSize, sizeof(int));

	for (int i = 0; i < nDocSize; i++)
	{
		int nKP = 0;
		DBloader.read((char*)&nKP, sizeof(int));
		std::vector<cv::KeyPoint> kps;
		std::vector<cv::Vec3f> threeDs;
		for (int j = 0; j < nKP; j++)
		{
			cv::KeyPoint kp;
			DBloader.read((char*)&kp.angle, sizeof(float));
			DBloader.read((char*)&kp.class_id, sizeof(int));
			DBloader.read((char*)&kp.octave, sizeof(int));
			DBloader.read((char*)&kp.pt.x, sizeof(float));
			DBloader.read((char*)&kp.pt.y, sizeof(float));
			DBloader.read((char*)&kp.response, sizeof(float));
			DBloader.read((char*)&kp.size, sizeof(float));
			kps.push_back(kp);

			cv::Vec3f threeD;
			DBloader.read((char*)&threeD.val[0], sizeof(float));
			DBloader.read((char*)&threeD.val[1], sizeof(float));
			DBloader.read((char*)&threeD.val[2], sizeof(float));
			threeDs.push_back(threeD);
		}

		int dscr_cols = 0;
		int dscr_rows = 0;
		int dscr_type = 0;
		int dscr_nData = 0;

		DBloader.read((char*)&dscr_cols, sizeof(int));
		DBloader.read((char*)&dscr_rows, sizeof(int));
		DBloader.read((char*)&dscr_type, sizeof(int));

		if (dscr_rows != 0)
		{
			if (dscr_type == 0)
				dscr_nData = sizeof(uchar);
			else if (dscr_type == 5)
				dscr_nData = sizeof(float);
			else if (dscr_type == 6)
				dscr_nData = sizeof(double);
			else {
				m_vecDB.clear();
				return false;
			}
		}
		else {
			m_vecDB.clear();
			return false;
		}
		cv::Mat tmpDscr(dscr_rows, dscr_cols, dscr_type);
		DBloader.read((char*)tmpDscr.data, dscr_nData * dscr_cols * dscr_rows);
		
		DB db;
		db.m_vecDescriptor = tmpDscr;
		db.m_vecKeyPoint = kps;
		db.m_vecWorldCoord = threeDs;
		m_vecDB.push_back(db);
	}

	DBloader.close();
	return true;
}

void Stereo::keyframe()
{
	DB db;
	db.m_vecDescriptor = m_output.m_Descriptor;
	db.m_vecKeyPoint = m_output.m_KeyPoint;
	db.m_vecWorldCoord = m_output.m_WorldCoord;

	m_vecDB.push_back(db);
}
