#include "CRPatch.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <deque>



void CRPatch::extractPatches(Mat &img, Mat &depth_img, unsigned int n, int label, int imageID, Rect &box, Point &vCenter, float scale) {
	// extract features
	vector<Mat> vImg;
	extractFeatureChannels(img, depth_img, vImg, scale);

	int offx = width / 2;
	int offy = height / 2;

	// generate x,y locations
	CvMat *locations = cvCreateMat( (img.cols - width) * (img.rows - height), 1, CV_32SC2 );
	if (box.width == 0)
		cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(0, 0, 0, 0), cvScalar(img.cols - width, img.rows - height, 0, 0) );
	else
		cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(box.x, box.y, 0, 0), cvScalar(box.x + box.width - width, box.y + box.height - height, 0, 0) );

	// reserve memory
	unsigned int offset = vLPatches[label].size();
	vLPatches[label].reserve(offset + n);
	for (unsigned int i = 0; i < n ; ++i) {

		CvPoint pt = *(CvPoint *)cvPtr1D( locations, i, 0 );

		PatchFeature pf;
		vLPatches[label].push_back(pf);
		vImageIDs[label].push_back(imageID);// adding the image id to the patch


		vLPatches[label].back().roi.x = pt.x;  vLPatches[label].back().roi.y = pt.y;
		vLPatches[label].back().roi.width = width;  vLPatches[label].back().roi.height = height;

		vLPatches[label].back().center.x = pt.x + offx - vCenter.x;
		vLPatches[label].back().center.y = pt.y + offy - vCenter.y;

		vLPatches[label].back().vPatch.resize(vImg.size());
		for (unsigned int c = 0; c < vImg.size(); ++c) {
			vLPatches[label].back().vPatch[c] = vImg[c](vLPatches[label].back().roi).clone();
		}

	}
}



void CRPatch::extractFeatureChannels(Mat &img, Mat &depth_img, std::vector<Mat> &vImg, float scale) {
	// 32 feature channels
	// 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)

	vImg.resize(58);
	for (unsigned int c = 0; c < vImg.size(); ++c)
		vImg[c] = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);

	// Get intensity
	cvtColor( img, vImg[0], CV_RGB2GRAY );

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	Mat I_x, I_y;

	// |I_x|, |I_y|
	Sobel(vImg[0], I_x, CV_16SC1, 1, 0, 3);
	Sobel(vImg[0], I_y, CV_16SC1, 0, 1, 3);
	convertScaleAbs( I_x, vImg[3], 0.25);
	convertScaleAbs( I_y, vImg[4], 0.25);

	{
		// Orientation of gradients
		for (int y = 0; y < img.rows; ++y) {
			short *dataX = I_x.ptr<short>(y);
			short *dataY = I_y.ptr<short>(y);
			uchar *dataZ = vImg[1].ptr<uchar>(y);

			for (int x = 0; x < img.cols; ++x) {
				// Avoid division by zero
				float tx = dataX[x] + _copysign(0.000001f, (float)dataX[x]);
				// Scaling [-pi/2 pi/2] -> [0 80*pi]
				dataZ[x] = uchar( ( atan((float)dataY[x] / tx) + 3.14159265f / 2.0f ) * 80 );
			}
		}
	}


	{
		// Magnitude of gradients
		for (int y = 0; y < img.rows; ++y) {
			short *dataX = I_x.ptr<short>(y);
			short *dataY = I_y.ptr<short>(y);
			uchar *dataZ = vImg[2].ptr<uchar>(y);

			for (int x = 0; x < img.cols; ++x) {
				dataZ[x] = (uchar)( sqrt(float(dataX[x] * dataX[x] + dataY[x] * dataY[x])) );
			}
		}
	}

	// 9-bin HOG feature stored at vImg[7] - vImg[15]
	vector<Mat> vImgHog(vImg.begin() + 7, vImg.begin() + 7 + 9);
	hog.extractOBin(vImg[1], vImg[2], vImgHog);

	// |I_xx|, |I_yy|
	Sobel(vImg[0], I_x, CV_16SC1, 2, 0, 3);
	Sobel(vImg[0], I_y, CV_16SC1, 0, 2, 3);

	convertScaleAbs( I_x, vImg[5], 0.25);
	convertScaleAbs( I_y, vImg[6], 0.25);

	// L, a, b
	cvtColor(img, img, CV_RGB2Lab);
	split(img, vector<Mat>(vImg.begin(), vImg.begin() + 3));


	// depth image
	Mat depth_img_mat(depth_img);
	Mat dI_x, dI_y;
	Mat tmp;

	convertScaleAbs(depth_img_mat / scale, vImg[17], 0.075);

	Sobel(depth_img_mat, dI_x, CV_32FC1, 2, 0, 3);
	Sobel(depth_img_mat, dI_y, CV_32FC1, 0, 2, 3);
	dI_x /= 16.0;
	dI_y /= 16.0;

	dI_x = abs(dI_x);
	dI_x *= 5;
	threshold(dI_x, dI_x, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(dI_x, vImg[18], 1);
	dI_y = abs(dI_y);
	dI_y *= 5;
	threshold(dI_y, dI_y, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(dI_y, vImg[19], 1);


	// min filter
	for (int c = 0; c < 29; ++c)
		erode(vImg[c], vImg[c + 29], Mat(5, 5, CV_8UC1));

	// max filter
	for (int c = 0; c < 29; ++c)
		dilate(vImg[c], vImg[c], Mat(5, 5, CV_8UC1));
}