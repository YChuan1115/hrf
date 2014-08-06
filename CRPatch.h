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

// structure for image patch
struct PatchFeature {
	PatchFeature() {}

	CvRect roi;
	CvPoint center;

	std::vector<CvMat*> vPatch;
	void print() const {
		std::cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height;
		std::cout << " " << center.x << " " << center.y; std::cout << std::endl;
	}
	void show(int delay) const;
};

static HoG hog; 

class CRPatch {
public:
	CRPatch(CvRNG* pRNG, int w, int h) : cvRNG(pRNG), width(w), height(h) {}
	void setClasses(int l) {vLPatches.resize(l); vImageIDs.resize(l);}

	// Extract patches from image
	void extractPatches(IplImage *img, IplImage *depth_img, unsigned int n, int label, CvRect* box = 0, CvPoint* vCenter = 0, float scale = 1.0);
	// Extract patches from image and adding its id to the patch (in vImageIDs)
	void extractPatches(IplImage *img, IplImage *depth_img, unsigned int n, int label, int imageID, CvRect* box = 0, CvPoint* vCenter = 0, float scale = 1.0);

	// Extract features from image
	static void extractFeatureChannels(IplImage *img, IplImage *depth_img, std::vector<IplImage*>& vImg, float scale);

	// min/max filter
	static void maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(IplImage *src, unsigned int width);
	static void maxfilt(IplImage *src, IplImage *dst, unsigned int width);
	static void minfilt(IplImage *src, unsigned int width);
	static void minfilt(IplImage *src, IplImage *dst, unsigned int width);

	std::vector<std::vector<PatchFeature> > vLPatches;
	std::vector<std::vector<int> > vImageIDs;// vector the same size as
private:
	CvRNG *cvRNG;
	int width;
	int height;
};

