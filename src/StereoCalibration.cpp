#include "StereoCalibration.h"

StereoCalibration::StereoCalibration(int board_width, int board_height, float square_width, float square_height)
	: boardSize(board_width, board_height), squareSize(square_width, square_height)
{
	m_calibFlag = false;
	m_cameraMatrix1 = Mat::eye(3, 3, CV_64F);
	m_cameraMatrix2 = Mat::eye(3, 3, CV_64F);
	m_distCoeffs1 = Mat::zeros(8, 1, CV_64F);
	m_distCoeffs2 = Mat::zeros(8, 1, CV_64F);
}

void StereoCalibration::InitCalibration()
{
	m_imagePoints1.clear();
	m_imagePoints2.clear();
}

int StereoCalibration::FindChessboard(Mat &view1, Mat &view2, bool reg)
{
	//assert (view1.size() == view2.size());
	m_imageSize = view1.size();

	Mat viewGray1, viewGray2;
	cvtColor(view1, viewGray1, CV_BGR2GRAY);
	cvtColor(view2, viewGray2, CV_BGR2GRAY);

	vector<Point2f> pointbuf1, pointbuf2;
	bool found1 = findChessboardCorners(view1, boardSize, pointbuf1,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
	bool found2 = findChessboardCorners(view2, boardSize, pointbuf2,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

	if (found1 && found2) {
		// improve the found corners' coordinate accuracy
		cornerSubPix(viewGray1, pointbuf1, Size(1, 1), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cornerSubPix(viewGray2, pointbuf2, Size(1, 1), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

		if (reg) {
			m_imagePoints1.push_back(pointbuf1);
			m_imagePoints2.push_back(pointbuf2);
		}
		drawChessboardCorners(view1, boardSize, Mat(pointbuf1), found1);
		drawChessboardCorners(view2, boardSize, Mat(pointbuf2), found2);
	}
	return m_imagePoints1.size();
}

bool StereoCalibration::RunCalibration()
{
	vector<vector<Point3f> > objectPoints(1);

	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++) {
			objectPoints[0].push_back(Point3f(j*squareSize.width, i*squareSize.height, 0.f));
		}
	}
	objectPoints.resize(m_imagePoints1.size(), objectPoints[0]);

	//
	// 좌우 camera의 체스판 영상의 점들로부터 camera matrix, distortion coefficients와 R, P 행렬을 계산한다
	//

	Mat E, F;
	//double rms = stereoCalibrate (objectPoints, m_imagePoints1, m_imagePoints2, 
	//	m_cameraMatrix1, m_distCoeffs1, m_cameraMatrix2, m_distCoeffs2,
	//	m_imageSize, m_R, m_T, m_E, m_F,										// R: rotation matrix, T: translation vector, E: Essential Matrix, F: Fundamental Matrix
	//	TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
	//	CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_SAME_FOCAL_LENGTH | 
	//	CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	double rms = stereoCalibrate(objectPoints, m_imagePoints1, m_imagePoints2,
		m_cameraMatrix1, m_distCoeffs1, m_cameraMatrix2, m_distCoeffs2,
		m_imageSize, m_R, m_T, m_E, m_F,
		CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_SAME_FOCAL_LENGTH |
		CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	bool ok = checkRange(m_cameraMatrix1) && checkRange(m_distCoeffs1) && checkRange(m_cameraMatrix2) && checkRange(m_distCoeffs2);
	if (ok) {
		Rect validRoi1, validRoi2;

		// Bouguet Algorithm
		stereoRectify(m_cameraMatrix1, m_distCoeffs1, m_cameraMatrix2, m_distCoeffs2,
			m_imageSize, m_R, m_T, m_R1, m_R2, m_P1, m_P2, m_Q, CALIB_ZERO_DISPARITY, 1, m_imageSize, &validRoi1, &validRoi2);

		initUndistortRectifyMap(m_cameraMatrix1, m_distCoeffs1, m_R1, m_P1, m_imageSize, CV_16SC2, map1, map2);
		initUndistortRectifyMap(m_cameraMatrix2, m_distCoeffs2, m_R2, m_P2, m_imageSize, CV_16SC2, map3, map4);
	}
	m_calibFlag = ok;
	return ok;
}

bool StereoCalibration::LoadCalibrationData()
{
	m_calibFlag = false;
	// reading intrinsic parameters
	FileStorage fs("intrinsics.yml", CV_STORAGE_READ);
	if (!fs.isOpened()) return false;

	fs["M1"] >> m_cameraMatrix1;
	fs["D1"] >> m_distCoeffs1;
	fs["M2"] >> m_cameraMatrix2;
	fs["D2"] >> m_distCoeffs2;

	//read extrinsic parameters
	Rect validRoi1, validRoi2;

	fs.open("extrinsics.yml", CV_STORAGE_READ);
	if (!fs.isOpened()) return false;

	fs["R"] >> m_R;
	fs["T"] >> m_T;
	fs["R1"] >> m_R1;
	fs["P1"] >> m_P1;
	fs["R2"] >> m_R2;
	fs["P2"] >> m_P2;
	fs["Q"] >> m_Q;
	FileNode is = fs["imageSize"];
	m_imageSize.width = is[0];
	m_imageSize.height = is[1];

	initUndistortRectifyMap(m_cameraMatrix1, m_distCoeffs1, m_R1, m_P1, m_imageSize, CV_16SC2, map1, map2);
	initUndistortRectifyMap(m_cameraMatrix2, m_distCoeffs2, m_R2, m_P2, m_imageSize, CV_16SC2, map3, map4);

	m_calibFlag = true;
	return true;
}

void StereoCalibration::Undistort(const Mat &view1, Mat &rview1, const Mat &view2, Mat &rview2)
{
	if (/*map1.data && map2.data && map3.data && map4.data*/m_calibFlag) {
		remap(view1, rview1, map1, map2, INTER_LINEAR);
		remap(view2, rview2, map3, map4, INTER_LINEAR);
	}
}


bool StereoCalibration::SaveCalibrationData()
{
	if (!m_calibFlag) {
		cout << "don't have calibration data\n";
		return false;
	}
	// save intrinsic parameters
	FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
	if (!fs.isOpened()) return false;

	fs << "M1" << m_cameraMatrix1;
	fs << "D1" << m_distCoeffs1;
	fs << "M2" << m_cameraMatrix2;
	fs << "D2" << m_distCoeffs2;
	fs.release();

	fs.open("extrinsics.yml", CV_STORAGE_WRITE);
	if (!fs.isOpened()) return false;

	fs << "R" << m_R;
	fs << "T" << m_T;
	fs << "R1" << m_R1;
	fs << "R2" << m_R2;
	fs << "P1" << m_P1;
	fs << "P2" << m_P2;
	fs << "Q" << m_Q;
	fs << "imageSize" << m_imageSize;
	fs.release();

	return true;
}

