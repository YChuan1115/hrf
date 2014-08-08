#pragma once

#include "Candidate.h"

using namespace cv;


Candidate::Candidate(const CRForest *crForest, Mat &img, std::vector<float> candidate, int candNr, bool do_bpr) {

	bpr = do_bpr;

	weight = candidate[0];
	x = candidate[1];
	y = candidate[2];
	scale = candidate[3];
	c = int(candidate[4]);
	r = int(candidate[5]);

	id = candNr;

	//x_img and s_img should be already rescaled
	n_trees = int(crForest->vTrees.size());
	n_classes = int(crForest->GetNumLabels());

	if (bpr) {
		// initialize the backprojection mask
		backproj_mask = Mat::zeros(img.cols * scale + 0.5, img.rows * scale + 0.5, CV_32FC1);
	}

}


void Candidate::getBBfromBpr(int thresh, bool do_sym) {
	// structure for storing the bounding box
	bb.resize(4);// MIN_X MIN_Y MAX_X MAX_Y

	// initialize the thresholded image
	Mat mask_thresh = Mat::zeros(backproj_mask.cols, backproj_mask.rows, CV_8UC1);

	// make the smoothed image
	Mat backproj_mask_smooth = backproj_mask.clone();
	blur(backproj_mask, backproj_mask_smooth, Size(9, 9));

	// get the maximum and minimum values in the backprojection mask
	double min_val_temp = 0;
	double max_val_temp = 0;
	minMaxLoc(backproj_mask_smooth, &min_val_temp, &max_val_temp);

	// determine the threshold
	double thresh_val = float(thresh) * (min_val_temp + max_val_temp) / 20.0f ; // TODO: make this a separate parameter

	// thresholding the image
	threshold( backproj_mask_smooth, mask_thresh, thresh_val, 1, CV_THRESH_BINARY);

	// now we have to determine the box around the image
	int min_x = backproj_mask.cols, min_y = backproj_mask.rows, max_x = -1, max_y = -1;

	for (int y_ind = 0; y_ind < mask_thresh.rows ; ++y_ind) {
		uchar *ptr = mask_thresh.ptr<uchar>(y_ind);
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
		float half_height = std::max(y - min_y + 1, max_y - y + 1);
		float half_width = std::max(x - min_x + 1, max_x - x + 1);
		min_x = x - half_width;
		max_x = x + half_width;
		min_y = y - half_height;
		max_y = y + half_height;
	}

	bb[0] = min_x;
	bb[1] = min_y;
	bb[2] = max_x;
	bb[3] = max_y;
}


void Candidate::clear() {
	if (bpr) {
		backproj_mask.release();
	}
}