#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <boost/progress.hpp>

#include "HoG.cpp"

using namespace std;
using namespace cv;

#define PATH_SEP "/"

//18 0 0 1 16 3 0 0 17 

int main(int argc, char const *argv[]) {
	vector<Mat> hogVector(9);
	vector<Mat> channels;
	HoG hog;
	Mat img = imread("/home/stfn/dev/rgbd-dataset/rgbd-dataset/cereal_box/cereal_box_1/cereal_box_1_1_1_crop.png");
	split(img, channels);
	{
		boost::progress_timer t;
		for (int i = 0; i < 100; ++i) {
			for (int i = 0; i < 9; ++i) {
				hogVector[i] = Mat::zeros(img.cols, img.rows, CV_8UC1);
			}
			hog.extractOBin(channels[0], channels[1], hogVector);
		}

	}
	for (int i = 0; i < 9; ++i) {
		cout << (int)hogVector[i].at<uchar>(10, 10) << " ";
	}
	cout << endl;

	return 0;



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