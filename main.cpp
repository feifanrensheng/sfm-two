#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include <Eigen/Geometry> 
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace cv;
using namespace std;

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
	)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	//¶ÁÈ¡Í¼Ïñ£¬»ñÈ¡Í¼ÏñÌØÕ÷µã£¬²¢±£´æ
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty()) continue;

		vector<KeyPoint> key_points;
		Mat descriptor;
		//Å¼¶û³öÏÖÄÚ´æ·ÖÅäÊ§°ÜµÄ´íÎó
		sift->detectAndCompute(image, noArray(), key_points, descriptor);

		//ÌØÕ÷µã¹ýÉÙ£¬ÔòÅÅ³ý¸ÃÍ¼Ïñ
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = image.at<Vec3b>(p.y, p.x);
		}
		colors_for_all.push_back(colors);
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	//»ñÈ¡Âú×ãRatio TestµÄ×îÐ¡Æ¥ÅäµÄ¾àÀë
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//ÅÅ³ý²»Âú×ãRatio TestµÄµãºÍÆ¥Åä¾àÀë¹ý´óµÄµã
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//±£´æÆ¥Åäµã
		matches.push_back(knn_matches[r][0]);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//¸ù¾ÝÄÚ²Î¾ØÕó»ñÈ¡Ïà»úµÄ½¹¾àºÍ¹âÐÄ×ø±ê£¨Ö÷µã×ø±ê£©
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//¸ù¾ÝÆ¥ÅäµãÇóÈ¡±¾Õ÷¾ØÕó£¬Ê¹ÓÃRANSAC£¬½øÒ»²½ÅÅ³ýÊ§Åäµã
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//¶ÔÓÚRANSAC¶øÑÔ£¬outlierÊýÁ¿´óÓÚ50%Ê±£¬½á¹ûÊÇ²»¿É¿¿µÄ
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//·Ö½â±¾Õ÷¾ØÕó£¬»ñÈ¡Ïà¶Ô±ä»»
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//Í¬Ê±Î»ÓÚÁ½¸öÏà»úÇ°·½µÄµãµÄÊýÁ¿Òª×ã¹»´ó
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}

void get_matched_points(
	vector<KeyPoint>& p1, 
	vector<KeyPoint>& p2, 
	vector<DMatch> matches, 
	vector<Point2f>& out_p1, 
	vector<Point2f>& out_p2
	)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
	//Á½¸öÏà»úµÄÍ¶Ó°¾ØÕó[R T]£¬triangulatePointsÖ»Ö§³ÖfloatÐÍ
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
	proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);

	R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//Èý½ÇÖØ½¨
	triangulatePoints(proj1, proj2, p1, p2, structure);
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << structure.cols;
	
	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.cols; ++i)
	{
		Mat_<float> c = structure.col(i);
		c /= c(3);	//Æë´Î×ø±ê£¬ÐèÒª³ýÒÔ×îºóÒ»¸öÔªËØ²ÅÊÇÕæÕýµÄ×ø±êÖµ
		fs << Point3f(c(0), c(1), c(2));
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}

int main( int argc, char** argv )
{
	string img1 = "../0004.png";
	string img2 = "../0006.png";
	vector<string> img_names = { img1, img2 };

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<DMatch> matches;

	//±¾Õ÷¾ØÕó
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));

	//ÌáÈ¡ÌØÕ÷
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//ÌØÕ÷Æ¥Åä
	match_features(descriptor_for_all[0], descriptor_for_all[1], matches);

	//¼ÆËã±ä»»¾ØÕó
	vector<Point2f> p1, p2;
	vector<Vec3b> c1, c2;
	Mat R, T;	//Ðý×ª¾ØÕóºÍÆ½ÒÆÏòÁ¿
	Mat mask;	//maskÖÐ´óÓÚÁãµÄµã´ú±íÆ¥Åäµã£¬µÈÓÚÁã´ú±íÊ§Åäµã
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches, p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches, c1, c2);
	find_transform(K, p1, p2, R, T, mask);

	//ÈýÎ¬ÖØ½¨
	Mat structure;	//4ÐÐNÁÐµÄ¾ØÕó£¬Ã¿Ò»ÁÐ´ú±í¿Õ¼äÖÐµÄÒ»¸öµã£¨Æë´Î×ø±ê£©
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	reconstruct(K, R, T, p1, p2, structure);

	//±£´æ²¢ÏÔÊ¾
	vector<Mat> rotations = { Mat::eye(3, 3, CV_64FC1), R };
	vector<Mat> motions = { Mat::zeros(3, 1, CV_64FC1), T };
	maskout_colors(c1, mask);
	save_structure("../structure.yml", rotations, motions, structure, c1);

	//system(".\\Viewer\\SfMViewer.exe");
	// show point cloud model
	typedef pcl::PointXYZRGB PointT;
	typedef pcl::PointCloud<PointT> PointCloud;
	PointCloud::Ptr pointcloud( new PointCloud );
	for (int i = 0; i < structure.cols; ++i)
	{
		Mat_<float> col = structure.col(i);
		col /= col(3);
		PointT p;
		p.x = col(0);
		p.y = col(1);
		p.z = col(2);
		p.b = c1[i][0];
		p.g = c1[i][1];
		p.r = c1[i][2];
		pointcloud->points.push_back( p );
	}
	pointcloud->is_dense = false;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(pointcloud);

	int user_data;
	while(!viewer.wasStopped())
	{
		user_data++;
	}
	return 0;
}
