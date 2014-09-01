#include "Candidate.hpp"


Candidate::Candidate(CRForest::ConstPtr crForest, Mat &img, vector<float> candidate_data, int cand_id, bool do_bpr):
	do_bpr_(do_bpr),
	weight_(candidate_data[0]),
	x_(candidate_data[1]),
	y_(candidate_data[2]),
	scale_(candidate_data[3]),
	c_(candidate_data[4]),
	r_(candidate_data[5]),
	cand_id_(cand_id) {


	n_trees_ = int(crForest->vTrees_.size());
	n_classes_ = int(crForest->GetNumLabels());

	if (do_bpr_) {
		// initialize the backprojection mask
		backproj_mask_ = Mat::zeros(Size(img.cols * scale_ + 0.5, img.rows * scale_ + 0.5), CV_32FC1);
	}
}


void Candidate::getBBfromBpr(int thresh, bool do_sym) {
	bb_.resize(2);
	Mat mask_thresh, backproj_mask_smooth;

	// smooth backprojection mask
	blur(backproj_mask_, backproj_mask_smooth, Size(9, 9));

	// get the maximum and minimum values in the backprojection mask
	double min_val_temp = 0;
	double max_val_temp = 0;
	minMaxLoc(backproj_mask_smooth, &min_val_temp, &max_val_temp);

	// determine the threshold
	double thresh_val = float(thresh) * (min_val_temp + max_val_temp) / 20.0f ; // TODO: make this a separate parameter

	// thresholding the image
	threshold( backproj_mask_smooth, mask_thresh, thresh_val, 1, CV_THRESH_BINARY);

	// now we have to determine the box around the image
	int min_x = backproj_mask_.cols;
	int min_y = backproj_mask_.rows;
	int max_x = -1;
	int max_y = -1;

	for (int y_ind = 0; y_ind < mask_thresh.rows ; ++y_ind) {
		float *ptr = mask_thresh.ptr<float>(y_ind);
		for ( int x_ind = 0; x_ind < mask_thresh.cols; ++x_ind) {
			if (ptr[x_ind] > 0) {
				if (y_ind > max_y)
					max_y = y_ind;
				if (x_ind > max_x)
					max_x = x_ind;
				if (x_ind < min_x)
					min_x = x_ind;
				if (y_ind < min_y)
					min_y = y_ind;
			}
		}
	}

	// symmetrizing the box about the center if asked for
	if (do_sym) {
		float half_height = max(y_ - min_y + 1, max_y - y_ + 1);
		float half_width = max(x_ - min_x + 1, max_x - x_ + 1);
		min_x = x_ - half_width;
		max_x = x_ + half_width;
		min_y = y_ - half_height;
		max_y = y_ + half_height;
	}

	bb_[0].x = min_x;
	bb_[0].y = min_y;
	bb_[1].x = max_x;
	bb_[1].y = max_y;
}