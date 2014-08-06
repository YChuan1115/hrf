//
// C++ Implementation: Candidate
//
// Description: this class holds an structure for keeping the data for a candidate object hypothesis
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
//
//
//

#pragma once

#include "Candidate.h"


 Candidate::Candidate(const CRForest* crForest, IplImage* img, std::vector<float> candidate, int candNr, bool do_bpr){ 

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

	if (bpr){
		// initialize the backprojection mask
		backproj_mask = cvCreateImage(cvSize(int(img->width*scale+0.5),int(img->height*scale+0.5)), IPL_DEPTH_32F,1);
		cvSetZero(backproj_mask);
	}

}

void Candidate::save_bp( char* prefix){
	char backproj_name[3000];
	sprintf_s(backproj_name,"%s/backproj_%d.bin",prefix,id);

	//smoothing the backprojection before saving
	IplImage* backproj_mask_smooth = cvCloneImage(backproj_mask);
	cvSmooth(backproj_mask, backproj_mask_smooth, CV_BLUR, 9, 9);

	//saving
	saveFloatImage(backproj_name, backproj_mask_smooth);
	cvReleaseImage(&backproj_mask_smooth);
	
}

void Candidate::getBBfromBpr(int thresh, bool do_sym){
	// structure for storing the bounding box
	bb.resize(4);// MIN_X MIN_Y MAX_X MAX_Y

	// initialize the thresholded image
	IplImage* mask_thresh = cvCreateImage(cvSize(int(backproj_mask->width+0.5),int(backproj_mask->height+0.5)), IPL_DEPTH_8U,1);
	cvSetZero(mask_thresh);

	// make the smoothed image
	IplImage* backproj_mask_smooth = cvCloneImage(backproj_mask);
	cvSmooth(backproj_mask, backproj_mask_smooth, CV_BLUR, 9, 9);


	// get the maximum and minimum values in the backprojection mask
	CvPoint max_loc_temp;
	CvPoint min_loc_temp;
	double min_val_temp = 0;
	double max_val_temp = 0;
	cvMinMaxLoc(backproj_mask_smooth, &min_val_temp, &max_val_temp, &min_loc_temp, &max_loc_temp, NULL);

	// determine the threshold
	double threshold = float(thresh)*(min_val_temp+max_val_temp)/20.0f ;// TODO: make this a separate parameter

	// thresholding the image 
	cvThreshold( backproj_mask_smooth, mask_thresh, threshold, 1, CV_THRESH_BINARY);

	// we are done with the smoothed projection
	cvReleaseImage(&backproj_mask_smooth);

	// now we have to determine the box around the image
	int min_x=backproj_mask->width,min_y=backproj_mask->height,max_x=-1,max_y=-1;

	for (int y_ind =0; y_ind < mask_thresh->height ; y_ind++){
		uchar* ptr = (uchar*)(mask_thresh->imageData + y_ind*mask_thresh->widthStep);
		for ( int x_ind =0; x_ind < mask_thresh->width; x_ind++){
			if (ptr[x_ind] > 0){
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

	cvReleaseImage(&mask_thresh); 
	// symmetrizing the box about the center if asked for
	if (do_sym){
		float half_height = std::max(y-min_y+1,max_y-y+1);
		float half_width = std::max(x-min_x+1,max_x-x+1);
		min_x = x-half_width; 
		max_x = x+half_width;
		min_y = y-half_height;
		max_y = y+half_height;
	}
	
	bb[0] = min_x;
	bb[1] = min_y;
	bb[2] = max_x;
	bb[3] = max_y;	
}

void Candidate::clear(){
	if (bpr){
		cvReleaseImage(&backproj_mask);
	}
}



