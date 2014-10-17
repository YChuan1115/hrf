#include "CRPatch.hpp"
#include "GpuHoG.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>



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
	Mat I_x, I_y;
	vImg.resize(60);
	for (unsigned int c = 0; c < vImg.size(); ++c)
		vImg[c] = Mat::zeros(img.size(), CV_8UC1);


	// Get intensity
	cvtColor( img, vImg[0], CV_RGB2GRAY );

	// |I_x|, |I_y|
	Sobel(vImg[0], I_x, CV_16SC1, 1, 0, 3);
	Sobel(vImg[0], I_y, CV_16SC1, 0, 1, 3);
	convertScaleAbs( I_x, vImg[3], 0.25);
	convertScaleAbs( I_y, vImg[4], 0.25);

	// 9-bin HOG feature stored at vImg[7] - vImg[15]
	GpuHoG gpuHog;
	vector<Mat> vImgHog(vImg.begin() + 7, vImg.begin() + 7 + 9);
	gpuHog.compute(vImg[0], vImgHog);

	// |I_xx|, |I_yy|
	Sobel(vImg[0], I_x, CV_16SC1, 2, 0, 3);
	Sobel(vImg[0], I_y, CV_16SC1, 0, 2, 3);
	convertScaleAbs( I_x, vImg[5], 0.25);
	convertScaleAbs( I_y, vImg[6], 0.25);

	// L, a, b
	cvtColor(img, img, CV_RGB2Lab);
	split(img, vector<Mat>(vImg.begin(), vImg.begin() + 3));

	// min filter
	for (int c = 0; c < 16; ++c) {
		erode(vImg[c], vImg[c + 16], Mat(5, 5, CV_8UC1));
		dilate(vImg[c], vImg[c], Mat(5, 5, CV_8UC1));
	}



	// Depth HoG
	Sobel(depth_img, I_x, CV_32FC1, 1, 0, 7);
	Sobel(depth_img, I_y, CV_32FC1, 0, 1, 7);
	I_x /= 1280.f;
	I_y /= 1280.f;

	// depth gradient orientation and magnitude
	for (int y = 0; y < depth_img.rows; ++y) {
		float *grad_x = I_x.ptr<float>(y);
		float *grad_y = I_y.ptr<float>(y);
		uchar *grad_orient = vImg[32].ptr<uchar>(y);
		uchar *grad_mag = vImg[33].ptr<uchar>(y);

		for (int x = 0; x < depth_img.cols; ++x) {
			// Orientation of gradients
			float tx = grad_x[x] + _copysign(0.000001f, grad_x[x]);
			// Scaling [-pi pi] -> [0 80*pi]
			grad_orient[x] = uchar( (atan2(grad_y[x], grad_x[x]) + M_PI) * 40 );

			// Magnitude of gradients
			float mag = sqrt(grad_x[x]*grad_x[x] + grad_y[x]*grad_y[x]) * 5;
			grad_mag[x] = uchar( (mag > 255) ? 255 : mag );
		}
	}

	// 9-bin HOG feature stored at vImg[7] - vImg[15]
	vector<Mat> vImgDepthHog(vImg.begin() + 37, vImg.begin() + 37 + 9);
	hog.extractOBin(vImg[32], vImg[33], vImgDepthHog);


	// |dI_x|, |I_y|
	I_x = abs(I_x);
	I_x *= 5;
	threshold(I_x, I_x, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_x, vImg[33], 1);
	I_y = abs(I_y);
	I_y *= 5;
	threshold(I_y, I_y, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_y, vImg[34], 1);

	// |I_xx|, |I_yy|
	Sobel(depth_img, I_x, CV_32FC1, 2, 0, 7);
	Sobel(depth_img, I_y, CV_32FC1, 0, 2, 7);
	I_x /= 768;
	I_y /= 768;

	I_x = abs(I_x);
	I_x *= 5;
	threshold(I_x, I_x, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_x, vImg[35], 1);
	I_y = abs(I_y);
	I_y *= 5;
	threshold(I_y, I_y, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_y, vImg[36], 1);


	// scaled depth value
	float max = 5000.f;
	depth_img.convertTo(depth_img, CV_32FC1);

	for (size_t y = 0; y < depth_img.rows; ++y) {
		float *row_ptr = depth_img.ptr<float>(y);
		for (size_t x = 0; x < depth_img.cols; ++x) {
			if (row_ptr[x] > max)
				row_ptr[x] = max;
			else if (row_ptr[x] <= 0)
				row_ptr[x] == std::numeric_limits<uchar>::quiet_NaN();

		}
	}
	depth_img /= scale;
	cv::convertScaleAbs(depth_img, vImg[32], 255.0 / (max * 2.3333));


	// min filter
	for (int c = 32; c < 46; ++c) {
		erode(vImg[c], vImg[c + 14], Mat(5, 5, CV_8UC1));
		dilate(vImg[c], vImg[c], Mat(5, 5, CV_8UC1));
	}
}