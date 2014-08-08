/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
//
//  Modified: Nima Razavi, BIWI, ETH Zurich
//
*/

#include "CRPatch.h"
#include <highgui.h>

#include <deque>

using namespace std;


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
	Sobel(vImg[0], I_x, CV_16UC1, 1, 0, 3);
	Sobel(vImg[0], I_y, CV_16UC1, 0, 1, 3);
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
	vector<Mat> vImgHog(vImg.begin()+7, vImg.begin()+7+9);
	hog.extractOBin(vImg[1], vImg[2], vImgHog);

	// |I_xx|, |I_yy|
	Sobel(vImg[0], I_x, CV_16UC1, 2, 0, 3);
	Sobel(vImg[0], I_y, CV_16UC1, 0, 2, 3);

	convertScaleAbs( I_x, vImg[5], 0.25);
	convertScaleAbs( I_y, vImg[6], 0.25);

	// L, a, b
	cvtColor(img, img, CV_RGB2Lab);
	split(img, vector<Mat>(vImg.begin(), vImg.begin()+3));


	// depth image
	cv::Mat depth_img_mat(depth_img);
	cv::Mat dI_x, dI_y;
	cv::Mat tmp;

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
		minfilt(vImg[c], vImg[c + 29], 5);

	// max filter
	for (int c = 0; c < 29; ++c)
		maxfilt(vImg[c], 5);
}

void CRPatch::maxfilt(Mat &src, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		maxfilt(src.data + y*src.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		maxfilt(src.data + x*src.step[1], src.step[0], src.rows, width);

}

void CRPatch::maxfilt(Mat &src, Mat &dst, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		maxfilt(src.data + y*src.step[0], dst.data + y*dst.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		maxfilt(src.data + x*src.step[1], src.step[0], src.rows, width);

}

void CRPatch::minfilt(Mat &src, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		minfilt(src.data + y*src.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		minfilt(src.data + x*src.step[1], src.step[0], src.rows, width);

}

void CRPatch::minfilt(Mat &src, Mat &dst, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		minfilt(src.data + y*src.step[0], dst.data + y*dst.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		minfilt(src.data + x*src.step[1], src.step[0], src.rows, width);

}


void CRPatch::maxfilt(uchar *data, uchar *maxvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width + 1) / 2) * step;
	size *= step;
	width *= step;

	maxvalues[0] = data[0];
	for (unsigned int i = 0; i < d - step; i += step) {
		for (unsigned int k = i; k < d + i; k += step) {
			if (data[k] > maxvalues[i]) maxvalues[i] = data[k];
		}
		maxvalues[i + step] = maxvalues[i];
	}

	maxvalues[size - step] = data[size - step];
	for (unsigned int i = size - step; i > size - d; i -= step) {
		for (unsigned int k = i; k > i - d; k -= step) {
			if (data[k] > maxvalues[i]) maxvalues[i] = data[k];
		}
		maxvalues[i - step] = maxvalues[i];
	}

	deque<int> maxfifo;
	for (unsigned int i = step; i < size; i += step) {
		if (i >= width) {
			maxvalues[i - d] = data[maxfifo.size() > 0 ? maxfifo.front() : i - step];
		}

		if (data[i] < data[i - step]) {

			maxfifo.push_back(i - step);
			if (i ==  width + maxfifo.front())
				maxfifo.pop_front();

		} else {

			while (maxfifo.size() > 0) {
				if (data[i] <= data[maxfifo.back()]) {
					if (i ==  width + maxfifo.front())
						maxfifo.pop_front();
					break;
				}
				maxfifo.pop_back();
			}

		}

	}

	maxvalues[size - d] = data[maxfifo.size() > 0 ? maxfifo.front() : size - step];

}

void CRPatch::maxfilt(uchar *data, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width + 1) / 2) * step;
	size *= step;
	width *= step;

	deque<uchar> tmp;

	tmp.push_back(data[0]);
	for (unsigned int k = step; k < d; k += step) {
		if (data[k] > tmp.back()) tmp.back() = data[k];
	}

	for (unsigned int i = step; i < d - step; i += step) {
		tmp.push_back(tmp.back());
		if (data[i + d - step] > tmp.back()) tmp.back() = data[i + d - step];
	}


	deque<int> minfifo;
	for (unsigned int i = step; i < size; i += step) {
		if (i >= width) {
			tmp.push_back(data[minfifo.size() > 0 ? minfifo.front() : i - step]);
			data[i - width] = tmp.front();
			tmp.pop_front();
		}

		if (data[i] < data[i - step]) {

			minfifo.push_back(i - step);
			if (i ==  width + minfifo.front())
				minfifo.pop_front();

		} else {

			while (minfifo.size() > 0) {
				if (data[i] <= data[minfifo.back()]) {
					if (i ==  width + minfifo.front())
						minfifo.pop_front();
					break;
				}
				minfifo.pop_back();
			}

		}

	}

	tmp.push_back(data[minfifo.size() > 0 ? minfifo.front() : size - step]);

	for (unsigned int k = size - step - step; k >= size - d; k -= step) {
		if (data[k] > data[size - step]) data[size - step] = data[k];
	}

	for (unsigned int i = size - step - step; i >= size - d; i -= step) {
		data[i] = data[i + step];
		if (data[i - d + step] > data[i]) data[i] = data[i - d + step];
	}

	for (unsigned int i = size - width; i <= size - d; i += step) {
		data[i] = tmp.front();
		tmp.pop_front();
	}

}

void CRPatch::minfilt(uchar *data, uchar *minvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width + 1) / 2) * step;
	size *= step;
	width *= step;

	minvalues[0] = data[0];
	for (unsigned int i = 0; i < d - step; i += step) {
		for (unsigned int k = i; k < d + i; k += step) {
			if (data[k] < minvalues[i]) minvalues[i] = data[k];
		}
		minvalues[i + step] = minvalues[i];
	}

	minvalues[size - step] = data[size - step];
	for (unsigned int i = size - step; i > size - d; i -= step) {
		for (unsigned int k = i; k > i - d; k -= step) {
			if (data[k] < minvalues[i]) minvalues[i] = data[k];
		}
		minvalues[i - step] = minvalues[i];
	}

	deque<int> minfifo;
	for (unsigned int i = step; i < size; i += step) {
		if (i >= width) {
			minvalues[i - d] = data[minfifo.size() > 0 ? minfifo.front() : i - step];
		}

		if (data[i] > data[i - step]) {

			minfifo.push_back(i - step);
			if (i ==  width + minfifo.front())
				minfifo.pop_front();

		} else {

			while (minfifo.size() > 0) {
				if (data[i] >= data[minfifo.back()]) {
					if (i ==  width + minfifo.front())
						minfifo.pop_front();
					break;
				}
				minfifo.pop_back();
			}

		}

	}

	minvalues[size - d] = data[minfifo.size() > 0 ? minfifo.front() : size - step];

}

void CRPatch::minfilt(uchar *data, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width + 1) / 2) * step;
	size *= step;
	width *= step;

	deque<uchar> tmp;

	tmp.push_back(data[0]);
	for (unsigned int k = step; k < d; k += step) {
		if (data[k] < tmp.back()) tmp.back() = data[k];
	}

	for (unsigned int i = step; i < d - step; i += step) {
		tmp.push_back(tmp.back());
		if (data[i + d - step] < tmp.back()) tmp.back() = data[i + d - step];
	}


	deque<int> minfifo;
	for (unsigned int i = step; i < size; i += step) {
		if (i >= width) {
			tmp.push_back(data[minfifo.size() > 0 ? minfifo.front() : i - step]);
			data[i - width] = tmp.front();
			tmp.pop_front();
		}

		if (data[i] > data[i - step]) {

			minfifo.push_back(i - step);
			if (i ==  width + minfifo.front())
				minfifo.pop_front();

		} else {

			while (minfifo.size() > 0) {
				if (data[i] >= data[minfifo.back()]) {
					if (i ==  width + minfifo.front())
						minfifo.pop_front();
					break;
				}
				minfifo.pop_back();
			}

		}

	}

	tmp.push_back(data[minfifo.size() > 0 ? minfifo.front() : size - step]);

	for (unsigned int k = size - step - step; k >= size - d; k -= step) {
		if (data[k] < data[size - step]) data[size - step] = data[k];
	}

	for (unsigned int i = size - step - step; i >= size - d; i -= step) {
		data[i] = data[i + step];
		if (data[i - d + step] < data[i]) data[i] = data[i - d + step];
	}

	for (unsigned int i = size - width; i <= size - d; i += step) {
		data[i] = tmp.front();
		tmp.pop_front();
	}
}