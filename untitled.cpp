#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <boost/progress.hpp>

#include "HoG.cpp"

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {


	Mat img = imread("/home/stfn/dev/rgbd-dataset/rgbd-dataset/cereal_box/cereal_box_1/cereal_box_1_1_1_crop.png");
	Mat framed_img;
	copyMakeBorder(img, framed_img, 0, 30, 0, 50, BORDER_REPLICATE);

	imshow("f", framed_img);
	imshow("img", img);
	waitKey();



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