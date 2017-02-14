#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
	:akaze_thresh(3e-4), 
	K(cv::Matx33d(700.0, 0, 320.0, 0, 700.0, 240.0, 0, 0, 1.0)), nn_match_ratio(0.8), ransac_thresh(2.5)
{
	akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f, 5, 5);
	//akaze->setThreshold(akaze_thresh);

	matcher = cv::BFMatcher(cv::NORM_HAMMING);
}


FeatureExtractor::~FeatureExtractor()
{
}
void FeatureExtractor::pointFeatureExtracte(cv::Mat& src, std::vector<cv::KeyPoint>& kp, cv::Mat& dscr)
{
	kp.clear();
	if (src.empty())
	{
		printf("no image\n");
		return;
	}
	cv::Mat localRGB = src.clone();

	akaze->detectAndCompute(localRGB, cv::noArray(), kp, dscr);	
}

// Query, Train
bool FeatureExtractor::featureMatching(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& dscr1, cv::Mat& dscr2, double* matched_ratio)
{
	if (kp1.size() <= 0 || kp2.size() <= 0 || dscr1.rows <= 0 || dscr2.rows <= 0)
		return false;

	std::vector<std::vector<cv::DMatch>> matches;
	std::vector<cv::DMatch> matched_matches;
	std::vector<cv::KeyPoint> matched1, matched2;

	// Query, Train
	matcher.knnMatch(dscr1, dscr2, matches, 2);			// mather shared

	for (int i = 0; i < matches.size(); i++)
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance)
		{
			matched1.push_back(kp1[matches[i][0].queryIdx]);
			matched2.push_back(kp2[matches[i][0].trainIdx]);
			matched_matches.push_back(matches[i][0]);
		}

	cv::Mat inlier_mask, homography;
	std::vector<cv::KeyPoint> inliers1, inliers2;
	cv::Mat dscrInlier1, dscrInlier2;
	std::vector<cv::DMatch> inlier_matches;

	if (matched1.size() >= 4)
	{
		homography = cv::findHomography(Points(matched1), Points(matched2), cv::RANSAC, ransac_thresh, inlier_mask);
	}

	if (matched1.size() < 4 || homography.empty())
	{
		std::cout << "Not matche" << std::endl;
		return false;
	}

	std::vector<int> mappingIdx1;
	std::vector<int> mappingIdx2;
	for (int i = 0; i<matched1.size(); i++)
	{
		if (inlier_mask.at<uchar>(i))
		{
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			dscrInlier1.push_back(dscr1.row(matched_matches[i].queryIdx));
			dscrInlier2.push_back(dscr2.row(matched_matches[i].trainIdx));
			inlier_matches.push_back(cv::DMatch(new_i, new_i, 0));
			mappingIdx1.push_back(matched_matches[i].queryIdx);
			mappingIdx2.push_back(matched_matches[i].trainIdx);
		}
	}
	m_output.m_leftKp = inliers1;
	m_output.m_rightKp = inliers2;
	m_output.m_leftDescr = dscrInlier1.clone();
	m_output.m_rightDescr = dscrInlier2.clone();
	m_output.m_mappingIdx1 = mappingIdx1;
	m_output.m_mappingIdx2 = mappingIdx2;
	if(matched_ratio != nullptr)
		*matched_ratio = inliers1.size() / (double)matched1.size();

	return true;
}

void FeatureExtractor::allCompute()		// 초기버전
{
	std::vector<cv::KeyPoint> LKeyPt;
	std::vector<cv::KeyPoint> RKeyPt;
	cv::Mat LDesc, RDesc;

	if (m_input.m_LeftImg.empty() || m_input.m_RightImg.empty())
	{
		printf("no image\n");
		return;
	}
	cv::Mat left = m_input.m_LeftImg.clone();
	cv::Mat right = m_input.m_RightImg.clone();

	akaze->detectAndCompute(left, cv::noArray(), LKeyPt, LDesc);
	akaze->detectAndCompute(right, cv::noArray(), RKeyPt, RDesc);
	
	std::vector<std::vector<cv::DMatch>> matches;
	std::vector<cv::DMatch> matched_matches;
	std::vector<cv::KeyPoint> matched1, matched2;

	// Query, Train
	matcher.knnMatch(LDesc, RDesc, matches, 2);

	for (int i = 0; i < matches.size(); i++)
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance)
		{
			matched1.push_back(LKeyPt[matches[i][0].queryIdx]);
			matched2.push_back(RKeyPt[matches[i][0].trainIdx]);
			matched_matches.push_back(matches[i][0]);
		}
	
	cv::Mat inlier_mask, homography;
	std::vector<cv::KeyPoint> inliers1, inliers2;
	cv::Mat dscrInlier1, dscrInlier2;
	std::vector<cv::DMatch> inlier_matches;

	if (matched1.size() >= 4)
	{
		homography = cv::findHomography(Points(matched1), Points(matched2), cv::RANSAC, ransac_thresh, inlier_mask);
	}

	if (matched1.size() < 4 || homography.empty())
	{
		std::cout << "Not matche" << std::endl;
		return;
	}

	for (int i = 0; i<matched1.size(); i++)
	{
		if (inlier_mask.at<uchar>(i))
		{
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			dscrInlier1.push_back(LDesc.row(matched_matches[i].queryIdx));
			dscrInlier2.push_back(RDesc.row(matched_matches[i].trainIdx));
			inlier_matches.push_back(cv::DMatch(new_i, new_i, 0));
		}
	}
	m_output.m_leftKp = inliers1;
	m_output.m_rightKp = inliers2;
	m_output.m_leftDescr = LDesc;
	m_output.m_rightDescr = RDesc;

	double match_ratio = inliers1.size() * 1.0 / matched1.size();

	if (1) {
		cv::Mat ShowMatch;
		cv::hconcat(left, right, ShowMatch);
		std::vector<cv::Vec3b> colorMap = colorMapping((int)inliers1.size());
		m_output.m_color = colorMap;
		for (int i = 0; i < inliers1.size(); i++) {
			cv::Point pt1 = cv::Point((int)(inliers1[i].pt.x + 0.5f), (int)(inliers1[i].pt.y + 0.5f));
			cv::Point pt2 = cv::Point((int)(inliers2[i].pt.x + 640.5f), (int)(inliers2[i].pt.y + 0.5f));
			cv::Scalar color = cv::Scalar(colorMap[i].val[0], colorMap[i].val[1], colorMap[i].val[2]);
			cv::circle(ShowMatch, pt1, 3, color);
			cv::circle(ShowMatch, pt2, 3, color);
			cv::line(ShowMatch, pt1, pt2, color);
		}
		//cv::drawMatches(left, inliers1, right, inliers2, inlier_matches, ShowMatch);
		cv::imshow("Matches", ShowMatch);
		//cv::waitKey(10);
	}
}
std::vector<cv::Vec3b> FeatureExtractor::colorMapping(int Size)
{
	std::vector<cv::Vec3b> dst;

	double dHStep = 180.0 / (double)Size;

	for (int i = Size; i > 0; i--)
	{
		cv::Vec3b tmp;
		tmp.val[0] = (int)(i*dHStep);
		tmp.val[1] = 255;
		tmp.val[2] = 255;

		dst.push_back(tmp);
	}
	
	cv::cvtColor(dst, dst, CV_HSV2BGR);
	return dst;
}
void FeatureExtractor::lineCompute()
{
	std::vector<cv::Vec4f> L;

}

std::vector<cv::Point2f> FeatureExtractor::Points(std::vector<cv::KeyPoint> keyPoints)
{
	std::vector<cv::Point2f> res;
	for (unsigned i = 0; i < keyPoints.size(); i++)
		res.push_back(keyPoints[i].pt);
	return res;
}

void FeatureExtractor::run()
{
	//pointFeatureExtracte();
	std::vector<cv::KeyPoint> kp1, kp2;
	cv::Mat dscr1, dscr2;
	pointFeatureExtracte(m_input.m_LeftImg, kp1, dscr1);
	pointFeatureExtracte(m_input.m_RightImg, kp2, dscr2);

	double matched_ratio = 0;
	int64 time = cv::getTickCount();

	try
	{
		featureMatching(kp1, kp2, dscr1, dscr2, &matched_ratio);
	}
	catch (const std::exception&)
	{
		printf("no matcing\n");
		return;
	}

	time = cv::getTickCount() - time;
	double fps = (double)time / cv::getTickFrequency() * 1000.0;

	cv::Mat canvas;
	cv::hconcat(m_input.m_LeftImg, m_input.m_RightImg, canvas);
	std::vector<cv::Vec3b> colorMap = colorMapping((int)m_output.m_leftKp.size());
	m_output.m_color = colorMap;
	for (int i = 0; i < m_output.m_leftKp.size(); i++) {
		cv::Point pt1 = cv::Point((int)(m_output.m_leftKp[i].pt.x + 0.5f), (int)(m_output.m_leftKp[i].pt.y + 0.5f));
		cv::Point pt2 = cv::Point((int)(m_output.m_rightKp[i].pt.x + canvas.cols / 2.0 + 0.5f), (int)(m_output.m_rightKp[i].pt.y + 0.5f));
		cv::Scalar color = cv::Scalar(colorMap[i].val[0], colorMap[i].val[1], colorMap[i].val[2]);
		cv::circle(canvas, pt1, 3, color);
		cv::circle(canvas, pt2, 3, color);
		cv::line(canvas, pt1, pt2, color);
	}
	//cv::drawMatches(left, inliers1, right, inliers2, inlier_matches, ShowMatch);
	std::string str = "Inlier Ratio: " + std::to_string(matched_ratio) + ", FPS " + std::to_string(fps) + "ms";
	cv::putText(canvas, str, cv::Point(45, 45), cv::HersheyFonts(), 1, cv::Scalar(0, 255, 0), 2);
	cv::imshow("Matches2", canvas);
}
