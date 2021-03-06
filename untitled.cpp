#include "CRPatch.hpp"
#include "HoG.hpp"

#define timer fubar
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>

#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include "opencv2/gpu/gpu.hpp"

#include <functional>


using namespace std;
using namespace cv;


void gpu_minMaxLoc_test(Mat &img) {

	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);


	{
		boost::progress_timer t;
		gpu::GpuMat gpu_img(img_gray);
		double min, max;
		Point minP, maxP;
		for (int i = 0; i < 10000; ++i) {
			gpu::minMaxLoc(gpu_img, &min, &max, &minP, &maxP);
			gpu::GpuMat gpu_img_roi(gpu_img, Rect(maxP.x, maxP.y, 1, 1));
			gpu_img_roi = Scalar(0);
		}
		Mat img_back = Mat(gpu_img);
		cout << "gpu del roi: ";
	}

	{
		boost::progress_timer t;
		Mat img_clone = img_gray.clone();
		double min, max;
		Point minP, maxP;
		for (int i = 0; i < 10000; ++i) {
			minMaxLoc(img_clone, &min, &max, &minP, &maxP);
			Mat img_roi(img_clone, Rect(maxP.x, maxP.y, 1, 1));
			img_roi = Scalar(0);
		}
		Mat img_back = img_clone.clone();
		cout << "cpu del roi: ";
	}
}


void mat_access_test(Mat &img) {
	{
		boost::progress_timer t;
		for (int i = 0; i < 500; ++i) {
			Mat test(1000, 1000, CV_32FC1, Scalar(1));
			for (int y = 0; y < test.rows; ++y) {
				float *rowPtr = test.ptr<float>(y);
				for (int x = 0; x < test.cols; ++x) {
					rowPtr[x] += 5;
				}
			}
		}
	}
	{
		boost::progress_timer t;
		for (int i = 0; i < 500; ++i) {
			Mat test(1000, 1000, CV_32FC1, Scalar(1));
			for (int y = 0; y < test.rows; ++y) {
				for (int x = 0; x < test.cols; ++x) {
					test.ptr<float>(y)[x] += 5;
				}
			}
		}
	}
	{
		boost::progress_timer t;
		for (int i = 0; i < 500; ++i) {
			Mat test(1000, 1000, CV_32FC1, Scalar(1));
			for (int y = 0; y < test.rows; ++y) {
				for (int x = 0; x < test.cols; ++x) {
					test.at<float>(y, x) += 5;
				}
			}
		}
	}
}


int main(int argc, char const *argv[]) {

	gpu::printShortCudaDeviceInfo(gpu::getDevice());

	//Mat img = imread("/home/stfn/dev/rgbd-dataset/rgbd-dataset/cereal_box/cereal_box_1/cereal_box_1_1_1_crop.png");
	Mat img = imread("/home/stfn/dev/rgbd-dataset/rgbd-scenes/background/background_10/background_10_1.png");
	

	//gpu_minMaxLoc_test(img);

	//mat_access_test(img);
}