/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#define _copysign copysign

#include <cxcore.h>
#include <cv.h>

#include <vector>
#include <iostream>

#include "HoG.h"

using namespace std;
using namespace cv;



// structure for image patch
struct PatchFeature {
	PatchFeature() {}

	Rect roi;
	Point center;

	vector<Mat> vPatch;
	void print() const {
		cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height;
		cout << " " << center.x << " " << center.y; cout << endl;
	}
	void show(int delay) const;
};

static HoG hog;

class CRPatch {
public:
	CRPatch(CvRNG *pRNG, int w, int h) : cvRNG(pRNG), width(w), height(h) {}
	void setClasses(int l) {
		vLPatches.resize(l);
		vImageIDs.resize(l);
	}

	// Extract patches from image
	void extractPatches(Mat &img, Mat &depth_img, unsigned int n, int label, int imageID, Rect &rect, Point &vCenter, float scale = 1.0);
	// Extract features from image
	static void extractFeatureChannels(Mat &img, Mat &depth_img, vector<Mat> &vImg, float scale = 1.0);

	// min/max filter
	static void maxfilt(Mat &src, Mat &dst, unsigned int width);
	static void minfilt(Mat &src, Mat &dst, unsigned int width);
	static void minfilt(Mat &src, unsigned int width);
	static void maxfilt(Mat &src, unsigned int width);
	static void maxfilt(uchar *data, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar *data, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar *data, uchar *minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(uchar *data, uchar *maxvalues, unsigned int step, unsigned int size, unsigned int width);


	vector<vector<PatchFeature> > vLPatches;
	vector<vector<int> > vImageIDs;// vector the same size as

private:
	CvRNG *cvRNG;
	int width;
	int height;
};

