/*
// C++ Implementation CRForestDetector
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
*/
#pragma once

#include "CRForest.hpp"
#include "Candidate.hpp"

#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <vector>


using namespace std;
using namespace cv;

static vector<Mat> default_vImg__;
static Rect default_rect__;


class CRForestDetector {

private:
	CRForest::ConstPtr crForest_;
	vector<vector<int> > class_ids_;
	Size patch_size_;

public:
	// Constructor
	CRForestDetector(CRForest::ConstPtr pRF, int width, int height) : crForest_(pRF), patch_size_(width, height) {
		crForest_->GetClassID(class_ids_);
	}

	void detectPyramidMR(vector<vector<Mat> > &vImgAssign, vector<vector<float> > &candidates, vector<float> &scales, vector<float> &kernel_width, vector<float> &params, vector< vector<Mat> > &classProbs, Mat &depth_img);
	void fullAssignCluster(Mat &img, Mat &depth_img, vector<vector<Mat> > &vvImgAssign, vector<float> &scales);
	void getClassConfidence(vector<vector<Mat> > &vImgAssign, vector<vector<Mat> > &classConfidence);
	void getClassConfidencePerScale(vector<Mat> &vImgAssign, vector<Mat> &classConfidence, int nlabels);
	void voteForCandidate(Mat &depth_img, vector<Mat> &imgAssign, Candidate &new_cand, int kernel_width, float max_width, float max_height  );
	
	

private:
	void assignCluster(Mat &img, Mat &depth_img, vector<Mat> &vImgAssign, float scale);
	void detectPeaks(vector<vector<float> > &candidates, vector<vector<Mat> > &imgDetect, vector<float> scales, int max_cands, int kernel_width, float kernel_std, bool separate = true, float shift = -1.0f, int this_class = -1, float threshold = 0.0f);
	void voteColor(Mat &depth_img, vector<Mat> &vImgAssign, vector<Mat> &vImgDetect, vector<Mat> &classProbs, float xShift = -1.0f, float yShift = -1.0f, int this_class = -1, Rect &focus = default_rect__, float prob_threshold = -1);


/*************************************************************/

public:
	size_t GetNumLabels() const {
		return crForest_->GetNumLabels();
	}
	void GetClassID(vector<vector<int> > &v_class_ids) const {
		crForest_->GetClassID( v_class_ids);
	}

	/** returns false if it could not load the hierarchy */
	bool GetHierarchy(vector<HNode> &hierarchy) const {
		return crForest_->GetHierarchy(hierarchy);
	}

	CRForest::ConstPtr GetCRForest() const {
		return crForest_;
	}
};
