/*
// C++ Implementation CRForestDetector
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
*/
#pragma once

#include "CRForest.h"
#include "Candidate.h"

#include "utils.h"

#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;

static vector<Mat> default_vImg;
static long int default_int = -1;
static Rect default_rect;

class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest *pRF, int w, int h, bool bpr = true) : crForest(pRF), width(w), height(h), do_bpr(bpr) {
		crForest->GetClassID(Class_id);
	}

	// Detection functions
public:
	void detectPyramidMR(vector<vector<Mat> > &vImgAssign, vector<vector<float> > &candidates, vector<float> &scales, vector<float> &kernel_width, vector<float> &params, vector< vector<Mat> > &classProbs);

	void voteForCandidate(vector<Mat> &imgAssign, Candidate &new_cand, int kernel_width, float max_width, float max_height  );

	void getClassConfidence(vector<vector<Mat> > &vImgAssign, vector<vector<Mat> > &classConfidence);

	void getClassConfidencePerScale(vector<Mat> &vImgAssign, vector<Mat> &classConfidence, int nlabels);

	void fullAssignCluster(Mat &img, Mat &depth_img, vector<vector<Mat> > &vvImgAssign, vector<float> &scales);

private:
	void assignCluster(Mat &img, Mat &depth_img, vector<Mat> &vImgAssign, float scale);

	void voteColor(vector<Mat> &vImgAssign, vector<Mat> &vImgDetect, vector<Mat> &classProbs, float xShift = -1.0f, float yShift = -1.0f, int this_class = -1, Rect &focus = default_rect, float prob_threshold = -1);

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
