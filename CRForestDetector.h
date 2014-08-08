/*
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include <stdio.h>
#include <vector>
#include "CRForest.h"
#include "Candidate.h"
#include "utils.h"

using namespace std;
using namespace cv;

static vector<vector<Mat> > default_vvImg;
static vector<Mat> default_vImg;
static long int default_int = -1;
static Rect default_rect;

class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest *pRF, int w, int h, double s_points = -1.0 , double s_forest = -1.0, bool bpr = true) : crForest(pRF), width(w), height(h), sample_points(s_points), do_bpr(bpr) {
		crForest->GetClassID(Class_id);
	}

	// Detection functions
public:
	void detectPyramidMR(vector<vector<Mat> > &vImgAssign, vector<vector<float> > &candidates, vector<float> &scales, vector<float> &ratios, vector<float> kernel_width , int max_cands = 10, int this_class = -1, float threshold = 0.0f, float prob_threshold = -1.0f, vector< vector<Mat> > &classProbs = default_vvImg, long int &nvotes = default_int);

	void voteForCandidate(vector<Mat> &imgAssign, Candidate &new_cand, int kernel_width, vector<float> &ratios, float max_width, float max_height  );

	void getClassConfidence(vector<vector<Mat> > &vImgAssign, vector<vector<Mat> > &classConfidence);

	void fullAssignCluster(Mat &img, Mat &depth_img, vector<vector<Mat> > &vvImgAssign, vector<float> &scales);

private:
	void assignCluster(Mat &img, Mat &depth_img, vector<Mat> &vImgAssign, float scale);

	void voteColor(vector<Mat> &vImgAssign, vector<Mat> &vImgDetect, float xShift = -1.0f, float yShift = -1.0f, int this_class = -1, Rect &focus = default_rect, float prob_threshold = -1, vector<Mat> &classProbs = default_vImg, long int &nvotes = default_int );

	void detectPeaks(vector<vector<float> > &candidates, vector<vector<Mat> > &imgDetect, vector<float> scales, int max_cands, int kernel_width, float kernel_std, bool separate = true, float shift = -1.0f, int this_class = -1, float threshold = 0.0f);

	/*************************************************************/

public:
	// Get/Set functions
	unsigned int GetNumLabels() const {
		return crForest->GetNumLabels();
	}
	void GetClassID(vector<vector<int> > &v_class_ids) {
		crForest->GetClassID( v_class_ids);
	}
	bool GetHierarchy(vector<HNode> &hierarchy) { // returns false if it could not load the hierarchy
		return crForest->GetHierarchy(hierarchy);
	}

	const CRForest *GetCRForest() {
		return crForest;
	}
	//private variables
private:
	const CRForest *crForest;
	vector<vector<int> > Class_id;
	int width;
	int height;
	double sample_points;
	bool do_bpr;
};
