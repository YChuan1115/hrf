/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
//
//  Modified: Nima Razavi, BIWI, ETH Zurich
//
*/

#include "CRPatch.h"
#include "GpuHoG.cpp"

#include <highgui.h>

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
	// 58 feature channels

	gpu::GpuMat gpu_img(img);
	vImg.resize(58);
	vector<gpu::GpuMat> gpu_vImg(vImg.size());
	for (unsigned int c = 0; c < vImg.size(); ++c) {
		vImg[c] = Mat::zeros(img.size(), CV_8UC1);
		gpu_vImg[c] = gpu::GpuMat(img.size(), CV_8UC1);
	}

	// Get intensity
	gpu::cvtColor( gpu_img, gpu_vImg[0], CV_RGB2GRAY );

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	gpu::GpuMat gpu_I_x(img.size(), CV_32FC1);
	gpu::GpuMat gpu_I_y(img.size(), CV_32FC1);

	// |I_x|, |I_y|
	gpu::Sobel(gpu_vImg[0], gpu_I_x, CV_32FC1, 1, 0, 3);
	gpu::Sobel(gpu_vImg[0], gpu_I_y, CV_32FC1, 0, 1, 3);
	gpu_I_x.convertTo(gpu_vImg[3], CV_8UC1, 0.25);
	gpu_I_y.convertTo(gpu_vImg[4], CV_8UC1, 0.25);

	// gpu hog
	vector<gpu::GpuMat> gpu_vImgHog(gpu_vImg.begin() + 7, gpu_vImg.begin() + 16);
	GpuHoG gpuHog;
	gpuHog.compute(gpu_vImg[0], gpu_vImgHog);

	// |I_xx|, |I_yy|
	gpu::Sobel(gpu_vImg[0], gpu_I_x, CV_32FC1, 2, 0, 3);
	gpu::Sobel(gpu_vImg[0], gpu_I_y, CV_32FC1, 0, 2, 3);
	gpu_I_x.convertTo(gpu_vImg[5], CV_8UC1, 0.25);
	gpu_I_y.convertTo(gpu_vImg[6], CV_8UC1, 0.25);

	// L, a, b
	vector<gpu::GpuMat> gpu_lab_channels(gpu_vImg.begin(), gpu_vImg.begin() + 3);
	gpu::cvtColor(gpu_img, gpu_img, CV_RGB2Lab);
	gpu::split(gpu_img, gpu_lab_channels);


	// depth image
	gpu::GpuMat gpu_depth_img(depth_img);
	gpu_I_x.convertTo(gpu_vImg[17], CV_8UC1, (0.075/scale));

	gpu::Sobel(gpu_depth_img, gpu_I_x, CV_32FC1, 2, 0, 3);
	gpu::Sobel(gpu_depth_img, gpu_I_y, CV_32FC1, 0, 2, 3);
	gpu::divide(gpu_I_x, Scalar(16), gpu_I_x);
	gpu::divide(gpu_I_y, Scalar(16), gpu_I_y);

	gpu::abs(gpu_I_x, gpu_I_x);
	gpu::abs(gpu_I_y, gpu_I_y);

	gpu::multiply(gpu_I_x, Scalar(5), gpu_I_x);
	gpu::multiply(gpu_I_y, Scalar(5), gpu_I_y);

	gpu::threshold(gpu_I_x, gpu_I_x, 255, 255, CV_THRESH_TRUNC);
	gpu::threshold(gpu_I_y, gpu_I_y, 255, 255, CV_THRESH_TRUNC);

	gpu_I_x.convertTo(gpu_vImg[18], CV_8UC1);
	gpu_I_y.convertTo(gpu_vImg[19], CV_8UC1);

	gpu::Stream gpu_stream;
	Ptr<gpu::BaseFilter_GPU> bf_min = gpu::getMinFilter_GPU(CV_8UC1, CV_8UC1, Size(5, 5));
	Ptr<gpu::BaseFilter_GPU> bf_max = gpu::getMaxFilter_GPU(CV_8UC1, CV_8UC1, Size(5, 5));
	Ptr<gpu::FilterEngine_GPU> filter_min = gpu::createFilter2D_GPU(bf_min, CV_8UC1, CV_8UC1);
	Ptr<gpu::FilterEngine_GPU> filter_max = gpu::createFilter2D_GPU(bf_max, CV_8UC1, CV_8UC1);

	for (int c = 0; c < 29; ++c) {
		filter_min->apply(gpu_vImg[c], gpu_vImg[c+29]);
		filter_max->apply(gpu_vImg[c], gpu_vImg[c]);
		
		gpu_stream.enqueueDownload(gpu_vImg[c], vImg[c]);
		gpu_stream.enqueueDownload(gpu_vImg[c+29], vImg[c+29]);
	}

	gpu_stream.waitForCompletion();

	for (unsigned int c = 0; c < gpu_vImg.size(); ++c) {
		gpu_vImg[c].release();
	}
}

void CRPatch::maxfilt(Mat &src, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		maxfilt(src.data + y * src.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		maxfilt(src.data + x * src.step[1], src.step[0], src.rows, width);

}

void CRPatch::maxfilt(Mat &src, Mat &dst, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		maxfilt(src.data + y * src.step[0], dst.data + y * dst.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		maxfilt(src.data + x * src.step[1], src.step[0], src.rows, width);

}

void CRPatch::minfilt(Mat &src, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		minfilt(src.data + y * src.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		minfilt(src.data + x * src.step[1], src.step[0], src.rows, width);

}

void CRPatch::minfilt(Mat &src, Mat &dst, unsigned int width) {

	for (int  y = 0; y < src.rows; y++)
		minfilt(src.data + y * src.step[0], dst.data + y * dst.step[0], src.step[1], src.cols, width);

	for (int  x = 0; x < src.cols; x++)
		minfilt(src.data + x * src.step[1], src.step[0], src.rows, width);

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