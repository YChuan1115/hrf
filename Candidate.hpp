#pragma once

#include "CRForest.hpp"

#include <opencv2/core/core.hpp>

#include <vector>


using namespace cv;
using namespace std;



class Candidate {
public:
	float weight_;
	float x_; // width in this case
	float y_; // height in this case
	float scale_;
	int c_;//class
	int r_;//ratio
	int n_trees_;
	int n_classes_;
	int cand_id_;// candidate ID
	bool do_bpr_; // if the backprojection is held
	vector<Point2f> bb_; // bounding box structure
	Mat backproj_mask_; // backprojection image

public:
	Candidate(CRForest::ConstPtr crForest, Mat &img, vector<float> candidate, int cand_id, bool do_bpr = true);

	void getBBfromBpr(int thresh = 2, bool do_sym = false);
};
