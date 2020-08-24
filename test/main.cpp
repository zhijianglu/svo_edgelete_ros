// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <svo/getfile.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>

#include <svo/slamviewer.h>
#include<thread>

//ros
#include "ros/ros.h"
#include "sensor_msgs/image_encodings.h"
#include "ros/ros.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float64MultiArray.h"
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>

using namespace std;
using namespace cv;
namespace svo
{

void LoadImages(const string &strPathImg, vector<string> &vstrImageLeft, vector<double> &vTimeStamps)
{
	double freq = 1.0 / 30.0;  //输入dji数据目前不需要时间戳
	ifstream fTimes;
//    vector<string> v_img_files;

	if (getdir(strPathImg, vstrImageLeft) >= 0) {
		printf("found %d image files in folder %s!\n", (int)vstrImageLeft.size(), strPathImg.c_str());
	}
	else if (getFile(strPathImg, vstrImageLeft) >= 0) {
		printf("found %d image files in file %s!\n", (int)vstrImageLeft.size(), strPathImg.c_str());
	}
	else {
		printf("could not load file list! wrong path / file?\n");
	}
	int total_img = vstrImageLeft.size();
	vTimeStamps.reserve(total_img);

	string time_stamp;
	for (int i = 0; i < total_img; ++i) {
		time_stamp = vstrImageLeft[i];
		int start = max(time_stamp.find_last_of("-") + 1, time_stamp.find_last_of("/") + 1);
		time_stamp = time_stamp.substr(start, time_stamp.find_last_of(".") - start);
		double time_ = atof(time_stamp.c_str());
		cout << setprecision(20) << time_ << endl;
		vTimeStamps.push_back(time_);
	}

	//for icl nuim data:
	if (vTimeStamps[2] - vTimeStamps[0] >= 10) {
		vTimeStamps.clear();
		for (int i = 0; i < total_img; ++i) {
			vstrImageLeft[i] = strPathImg + "/" + to_string(i) + ".png";
			time_stamp = vstrImageLeft[i];
			int start = max(time_stamp.find_last_of("-") + 1, time_stamp.find_last_of("/") + 1);
			time_stamp = time_stamp.substr(start, time_stamp.find_last_of(".") - start);
			double time_ = atof(time_stamp.c_str());
			cout << setprecision(20) << time_ << endl;
			vTimeStamps.push_back(time_);
		}
	}
}

class BenchmarkNode
{
	svo::AbstractCamera *cam_;
	svo::AbstractCamera *cam_r_;
	svo::PinholeCamera *cam_pinhole_;
	svo::FrameHandlerMono *vo_;


	SLAM_VIEWER::Viewer *viewer_;
	std::thread *viewer_thread_;
	vector<string> vstrImg;
	vector<double> vTimeStamp;
	cv::Mat M1l;
	cv::Mat M2l;
public:
	BenchmarkNode(ros::NodeHandle& nh);
	~BenchmarkNode();
	void runFromFolder();
	void imageCallback(const sensor_msgs::ImageConstPtr &msg);
	ofstream pose_est;
	int image_cnt = 0;
	std_msgs::Header last_header;
	/*设置图像接受的节点*/
	ros::Publisher pose_pub;
	image_transport::Publisher img_pub;

	string image_input_topic;
	string image_pub_topic;
	string pose_pub_topic;

};

BenchmarkNode::BenchmarkNode(ros::NodeHandle& nh)
{

	cv::FileStorage fsSettings("/media/lab/E_Disk/Lab_Project/VO_modules/svo_edgelet_ws/src/svo_edgelet/config/air2.yaml", cv::FileStorage::READ);
//	cv::FileStorage fsSettings("/media/lab/E_Disk/Lab_Project/VO_modules/svo_edgelet_ws/src/svo_edgelet/config/dji47.yaml", cv::FileStorage::READ);
	if (!fsSettings.isOpened()) {
		cerr << "ERROR: Wrong path to settings" << endl;
		return;
	}

	fsSettings["image_input_topic"]>>image_input_topic;
	fsSettings["image_pub_topic"]>>image_pub_topic;
	fsSettings["pose_pub_topic"]>>pose_pub_topic;

	pose_est.open("pose_est.txt", ios::out);
	pose_pub = nh.advertise<geometry_msgs::PoseStamped>(pose_pub_topic, 100);
	image_transport::ImageTransport it(nh);
	img_pub = it.advertise(image_pub_topic, 1);


	int downsample_factor = fsSettings["downsample_factor"];

//	todo 获取内参
	double fx = fsSettings["cam_fx"];
	double fy = fsSettings["cam_fy"];
	double cx = fsSettings["cam_cx"];
	double cy = fsSettings["cam_cy"];

	double d0 = fsSettings["d0"];
	double d1 = fsSettings["d1"];
	double d2 = fsSettings["d2"];
	double d3 = fsSettings["d3"];
	double d4 = fsSettings["d4"];

	int cam_width = fsSettings["cam_width"];
	int cam_height = fsSettings["cam_height"];

	cv::Mat input_K = (cv::Mat_<float>(3, 3) <<
											 fx, 0.0f, cx,
		0.0f, fy, cy,
		0.0f, 0.0f, 1.0f);

	cv::Mat input_D;


	if (d4 != 0)
		input_D = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);
	else
		input_D = (cv::Mat_<float>(1, 4) << d0, d1, d2, d3);

	float resize_fx, resize_fy, resize_cx, resize_cy;

	resize_fx = (float)fx * (float)downsample_factor;
	resize_fy = (float)fy * (float)downsample_factor;
	resize_cx = (float)cx * (float)downsample_factor;
	resize_cy = (float)cy * (float)downsample_factor;

	cv::Mat resize_K;

	resize_K = (cv::Mat_<float>(3, 3) << resize_fx, 0.0f, resize_cx, 0.0f, resize_fy, resize_cy, 0.0f, 0.0f, 1.0f);

	cout << "resize_K: \n" << resize_K << endl;
	int resize_width = cam_width * downsample_factor;
	int resize_height = cam_height * downsample_factor;

	cv::initUndistortRectifyMap(
		input_K,
		input_D,
		cv::Mat_<double>::eye(3, 3),
		resize_K,
		cv::Size(resize_width, resize_height),
		CV_32F,
		M1l, M2l);

	cam_ = new svo::PinholeCamera(resize_width, resize_height, resize_fx, resize_fy, resize_cx, resize_cy);
	cam_r_ = new svo::PinholeCamera(resize_width, resize_height, resize_fx, resize_fy, resize_cx, resize_cy);

	vo_ = new svo::FrameHandlerMono(cam_);
	vo_->start();

	viewer_ = new SLAM_VIEWER::Viewer(vo_);
	viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run, viewer_);
	viewer_thread_->detach();

}

BenchmarkNode::~BenchmarkNode()
{
	delete vo_;
	delete cam_;
	delete cam_r_;
	delete cam_pinhole_;
	delete viewer_;
	delete viewer_thread_;
}

//#define TXTREAD
void BenchmarkNode::runFromFolder()
{
	const int nImages = vstrImg.size();
	cv::Mat imLeft;
//	int freq = 1.0/30.0;
	ofstream pose_est("pose_est.txt", ios::out);
//	pose_est.precision(18);
	for (int ni = 10; ni < nImages; ni++) {
		// Read left and right images from file
		imLeft = cv::imread(vstrImg[ni], CV_8UC1);

		assert(!imLeft.empty());
		cv::Mat imLeft_rect;
		cv::remap(imLeft, imLeft_rect, M1l, M2l, cv::INTER_LINEAR);
		cout << "img size:" << imLeft_rect.size() << endl;
		vo_->addImage(imLeft_rect, vTimeStamp[ni]);

		// display tracking quality
		if (vo_->lastFrame() != NULL) {
			std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
					  << "#Features: " << vo_->lastNumObservations() << " \n";
			// access the pose of the camera via vo_->lastFrame()->T_f_w_.

			if (vo_->lastFrame()->isKeyframe()) {

				//todo record pose
				Sophus::SE3 curr_pose = vo_->lastFrame()->T_f_w_.inverse();
				Eigen::Quaterniond q(curr_pose.rotation_matrix());

//				pose_est << (unsigned long)vTimeStamp[vo_->lastFrame()->id_] << " "  //for other data
				pose_est << to_string(vTimeStamp[vo_->lastFrame()->id_]) << " "      //for tum
						 << curr_pose.translation().x() << " "
						 << curr_pose.translation().y() << " "
						 << curr_pose.translation().z() << " "
						 << q.x() << " "
						 << q.y() << " "
						 << q.z() << " "
						 << q.w() << endl;
			}
			
		}
	}
	pose_est.close();
	cv::waitKey(0);
}

void BenchmarkNode::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
	cv::Mat imgRaw;
	cv::Mat imgInput;
//	pose_est.precision(18);

	// Read left and right images from file
	imgRaw = cv_bridge::toCvShare(msg, "bgr8")->image;
//	imgInput.convertTo(imgInput,CV_BGR2GRAY);
	cv::cvtColor(imgRaw, imgInput, CV_RGB2GRAY);


	assert(!imgInput.empty());

//	cv::Mat imLeft_rect;
	cv::remap(imgInput, imgInput, M1l, M2l, cv::INTER_LINEAR);//去畸变
//	cout << "img size:" << imgInput.size() << endl;

	vo_->addImage(imgInput, image_cnt);
//	cout<<"time stamp: "<<msg->header.stamp.toSec()<<endl;

//	imshow("imgInput", imgInput);
//	waitKey(1);

	// display tracking quality
	if (vo_->lastFrame() != NULL) {
		std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
				  << "#Features: " << vo_->lastNumObservations() << " \n";
		// access the pose of the camera via vo_->lastFrame()->T_f_w_.

		//只显示关键帧的图像
//		if (vo_->lastFrame()->isKeyframe() && vo_->lastFrame()->id_!=0) {
//			//todo record pose
//			Sophus::SE3 curr_pose = vo_->lastFrame()->T_f_w_.inverse();
//			Eigen::Quaterniond q(curr_pose.rotation_matrix());
//			pose_est << curr_pose.translation().x() << " "
//					 << curr_pose.translation().y() << " "
//					 << curr_pose.translation().z() << " "
//					 << q.x() << " "
//					 << q.y() << " "
//					 << q.z() << " "
//					 << q.w() << endl;
//		}

		Sophus::SE3 curr_pose = vo_->lastFrame()->T_f_w_.inverse();
		Eigen::Quaterniond q(curr_pose.rotation_matrix());
		pose_est << last_header.stamp.toSec() << " "
				 << curr_pose.translation().x() << " "
				 << curr_pose.translation().y() << " "
				 << curr_pose.translation().z() << " "
				 << q.x() << " "
				 << q.y() << " "
				 << q.z() << " "
				 << q.w() << endl;

		geometry_msgs::PoseStamped curr_pose_msg;

		curr_pose_msg.header = msg->header;
		curr_pose_msg.header.frame_id = "world";
		curr_pose_msg.pose.position.x = 5*curr_pose.translation().x();
		curr_pose_msg.pose.position.y = 5*curr_pose.translation().y();
		curr_pose_msg.pose.position.z = 5*curr_pose.translation().z();

		curr_pose_msg.pose.orientation.x = q.x();
		curr_pose_msg.pose.orientation.y = q.y();
		curr_pose_msg.pose.orientation.z = q.z();
		curr_pose_msg.pose.orientation.w = q.w();
		pose_pub.publish(curr_pose_msg);

		sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(msg->header, "mono8", imgInput).toImageMsg();
		img_pub.publish(img_msg);
	}

	last_header = msg->header;
	image_cnt++;
}
} // namespace svo


int main(int argc, char** argv)
{
	ros::init(argc, argv, "jetbot_ctrl");
	ros::NodeHandle nh;
	svo::BenchmarkNode* benchmark = new svo::BenchmarkNode(nh);
//	benchmark.runFromFolder();

	ros::Subscriber image_sub;
	std::string topic = benchmark->image_input_topic;

	cout<<"waiting for image from topic : "<<topic<<endl;

	image_sub = nh.subscribe(topic, 10, &svo::BenchmarkNode::imageCallback,benchmark);
	ros::spin();

	benchmark->pose_est.close();

	printf("BenchmarkNode finished.\n");

	return 0;
}

