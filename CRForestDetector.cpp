/*
// C++ Implementation CRForestDetector
//
// Description: The detector implementation.
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
*/
#pragma once

#include "CRForestDetector.h"
#include <vector>
#include <highgui.h>
#include <algorithm>


using namespace std;

int COUNT;

// given the cluster assignment images, we are voting into the voting space vImgDetect
void CRForestDetector::voteColor(vector<Mat> &vImgAssign, vector<Mat> &vImgDetect, float xShift, float yShift, int this_class, Rect &focus, float prob_threshold, vector<Mat> &classProbs, long int &nvotes) {
	// vImgDetect are all initialized before

	if (vImgAssign.size() < 1)
		return;
	bool do_count = true;
	if (nvotes < 0)
		do_count = false;

	// compensate for the shift based on the
	if (xShift < 0)
		xShift = vImgAssign[0].cols * 0.50;
	if (yShift < 0)
		yShift = vImgAssign[0].rows * 0.50;

	float ntrees = float(vImgAssign.size());


	// loop over trees
	for (unsigned int trNr = 0; trNr < vImgAssign.size(); ++trNr) {

		// loop over assign height
		for (unsigned int cy = 0 ; cy < vImgAssign[trNr].rows; ++cy) {
			float *ptr = vImgAssign[trNr].ptr<float>(cy);

			// loop over assign width
			for (unsigned int cx = 0; cx < vImgAssign[trNr].cols; ++cx) {
				// get the leaf_id
				if (ptr[cx] < 0)
					continue;

				LeafNode *tmp = crForest->vTrees[trNr]->getLeaf(ptr[cx]);

				// loop over labels
				for (unsigned int lNr = 0; lNr < vImgDetect.size(); ++lNr) {

					if ((this_class >= 0 ) && (this_class != lNr)) // the voting should be done on a single class only
						continue;

					bool condition;
					if (prob_threshold < 0) {
						condition = (Class_id[trNr][lNr] > 0 && tmp->vPrLabel[lNr] * Class_id[trNr].size() > 1);
					} else {
						condition = (Class_id[trNr][lNr] > 0  &&  classProbs[lNr].ptr<float>(cy)[cx] > prob_threshold);
					}

					if (condition) {
						// vote for all points stored in a leaf
						float w = tmp->vPrLabel[lNr] / ntrees;
						vector<float>::const_iterator itW = tmp->vCenterWeights[lNr].begin();
						for (vector<Point>::const_iterator it = tmp->vCenter[lNr].begin(); it != tmp->vCenter[lNr].end(); ++it, itW++) {
							// calc object hypothesis center
							int x = int(float(cx) - float((*it).x) + 0.5 + xShift);
							int y = int(float(cy) - float((*it).y) + 0.5 + yShift);

							if (do_count) {
								nvotes++;
							}

							// finally vote into voting space
							if (focus.width == 0) {
								if (y >= 0 && y < vImgDetect[lNr].rows && x >= 0 && x < vImgDetect[lNr].cols) {
									vImgDetect[lNr].ptr<float>(y)[x] += w * (*itW);
								}
							} else if (focus.contains(Point(x, y))) {
								vImgDetect[lNr].ptr<float>(y - focus.y)[x - focus.x] += w * (*itW);
							}
						}
					}
				}
			}
		}
	}
}


// Gathering the information in the support of each candidate
void CRForestDetector::voteForCandidate(vector<Mat> &vImgAssign, Candidate &cand, int kernel_width, vector<float> &ratios, float max_width, float max_height ) {

	if (vImgAssign.size() < 1)
		return;

	time_t t = time(NULL);
	int seed = (int)t;//1407685013;
	CvRNG pRNG(seed);
	double value = 0.0;
	double sample_votes = 0.998;// ignore the patches by this probability

	// get cand_x and cand_y at this scale
	int cand_x = int(cand.x * cand.scale + 0.5);
	int cand_y = int(cand.y * cand.scale + 0.5);

	int min_x, min_y, max_x, max_y;

	// initializing the box around the candidate center where the votes can come from
	min_x = cand_x - int(max_width / 2.0f + kernel_width + 0.5f);
	min_y = cand_y - int(max_height / 2.0f + kernel_width + 0.5f);
	min_x = max(min_x, 0);
	min_y = max(min_y, 0);

	max_x = cand_x + int(max_width / 2.0f + 0.5f) + 1;
	max_y = cand_y + int(max_height / 2.0f + 0.5f) + 1;
	max_x = min(vImgAssign[0].cols, max_x);
	max_y = min(vImgAssign[0].rows, max_y);


	// looping over all trees
	float ntrees = float(vImgAssign.size());
	for (int trNr = 0; trNr < int(ntrees); trNr++) {

		// looping over all locations within candidate roi
		for (unsigned int cy = min_y; cy < max_y; ++cy) {
			float *ptr = vImgAssign[trNr].ptr<float>(cy);

			for (unsigned int cx = min_x; cx < max_x; ++cx) {

				value = cvRandReal(&pRNG);
				
				if (value < sample_votes || ptr[cx] <0)
					continue;

				LeafNode *tmp = crForest->vTrees[trNr]->getLeaf(ptr[cx]);
				float w = tmp->vPrLabel[cand.c] / ntrees;
				if (w < 0.0e-7)
					continue;

				float w_element = 0.0f;
				int idNr = 0;
				vector<float>::const_iterator itW = tmp->vCenterWeights[cand.c].begin();
				for (vector<Point>::const_iterator it = tmp->vCenter[cand.c].begin() ; it != tmp->vCenter[cand.c].end(); ++it, ++idNr, itW++) {
					int x = int(float(cx) - float((*it).x) * ratios[cand.r] + 0.5);
					int y = int(float(cy) - float((*it).y) + 0.5);
					if (isInsideKernel2D(x, y, cand_x, cand_y , kernel_width)) {
						w_element += w * (*itW);
					}
				}

				if ( w_element > 0.0 && cand.bpr) {
					// update the backprojection image
					cand.backproj_mask.ptr<float>(cy)[cx] += w_element;
				}
			}
		}
	}
}



/********************************** FULL object detection ************************************/

void CRForestDetector::detectPeaks(vector<vector<float> > &candidates, vector<vector<Mat> > &imgDetect, vector<float> scales, int max_cands, int kernel_width, float kernel_std, bool separate, float shift, int this_class, float threshold) {

	candidates.clear();

	// this is just to access a non-empty detect image for getting sizes and so on
	int default_class = 0;
	if ((this_class >= 0) )
		default_class = this_class;

	// smoothing the accumulator matrix
	vector<vector<Mat> > smoothAcc;
	smoothAcc.resize(scales.size());
	for (unsigned int scNr = 0; scNr < scales.size(); ++scNr) {
		int adapKwidth = int(kernel_width * scales[scNr] / 2.0f) * 2 + 1;
		float adapKstd  = kernel_std * scales[scNr];

		smoothAcc[scNr].resize(imgDetect[scNr].size());
		for (unsigned int cNr = 0; cNr < imgDetect[scNr].size(); ++cNr) {
			if ((this_class >= 0) && ( this_class != cNr))
				continue;

			GaussianBlur(imgDetect[scNr][cNr], smoothAcc[scNr][cNr], Size(adapKwidth, adapKwidth), adapKstd);
		}
	}

	// each candidate is a six element vector weight, x, y, scale, class, ratio
	Point max_loc_temp;
	Point min_loc_temp;
	double min_val_temp = 0;
	double max_val_temp = 0;

	float xShift;
	float yShift;

	/***************** find the local maximum locations **********************/
	int candNr = 0;
	for (unsigned int count = 0; candNr < max_cands ; ++count) { // count can go until infinity
		bool flag = false;
		vector<float> max_position(6, -1); // max_val, x, y, scNr, cNr, rNr
		// detect the maximum
		for (unsigned int scNr = 0; scNr < scales.size(); ++scNr) {
			if (shift < 0.0f) {
				xShift = imgDetect[scNr][default_class].cols * 0.25;
				yShift = imgDetect[scNr][default_class].rows * 0.25;
			} else {
				xShift = imgDetect[scNr][default_class].cols * shift;
				yShift = imgDetect[scNr][default_class].rows * shift;
			}
			for (unsigned int cNr = 0; cNr < imgDetect[scNr].size(); ++cNr) {
				if ((this_class >= 0) && ( this_class != cNr))
					continue;

				minMaxLoc(smoothAcc[scNr][cNr], &min_val_temp, &max_val_temp, &min_loc_temp, &max_loc_temp);
				if ( (max_val_temp >= threshold) && (max_val_temp > max_position[0]) ) {
					flag = true;
					max_position[0] = max_val_temp;
					max_position[1] = float(-xShift + max_loc_temp.x) / scales[scNr];
					max_position[2] = float(-yShift + max_loc_temp.y) / scales[scNr];
					max_position[3] = scales[scNr];
					if (separate) {
						max_position[4] = cNr;
					} else {
						max_position[4] = -1;
					}
					max_position[5] = 0;
				}
			}
		}

		if (!flag)
			break;
		else
			candNr++;

		// push the candidate in the stack
		candidates.push_back(max_position);


		// remove the maximum region
		for (unsigned int scNr = 0; scNr < scales.size(); scNr++) {
			if (max_position[3] != scales[scNr]) {
				continue;
			}

			if (shift < 0.0f) {
				xShift = imgDetect[scNr][default_class].cols * 0.25;
				yShift = imgDetect[scNr][default_class].rows * 0.25;
			} else {
				xShift = imgDetect[scNr][default_class].cols * shift;
				yShift = imgDetect[scNr][default_class].rows * shift;
			}

			// remove the region with the supporting kernel width
			int adapHalfKwidth = int(kernel_width * (scales[scNr] / max_position[3])) + 1;
			float adapKstd  = kernel_std * scales[scNr];
			int cx = int(max_position[1] * scales[scNr] + xShift);
			int cy = int(max_position[2] * scales[scNr] + yShift);
			int x = max(0, cx - adapHalfKwidth);
			int y = max(0, cy - adapHalfKwidth);
			int rwidth = max(1, min(cx + adapHalfKwidth, smoothAcc[scNr][default_class].cols - 1) - x + 1);
			int rheight = max(1, min(cy + adapHalfKwidth, smoothAcc[scNr][default_class].rows - 1)  - y + 1);

			for (unsigned int cNr = 0; cNr < imgDetect[scNr].size(); ++cNr) {
				if (max_position[4] >= 0 && cNr != max_position[4])
					continue;

				// clear candidates bounding box
				smoothAcc[scNr][cNr](Rect(x, y, rwidth, rheight)) = Scalar(0.0);
			}
		}
	}
}

void CRForestDetector::detectPyramidMR(vector<vector<Mat> > &vImgAssign, vector<vector<float> > &candidates, vector<float> &scales, vector<float> &ratios, vector<float> kernel_width , int max_cands, int this_class, float threshold, float prob_threshold, vector<vector<Mat> > &classProbs, long int &nvotes) {


	vector<vector<Mat> > vvImgDetect(vImgAssign.size());

	// accumulating votes for all scales and classes
	for (unsigned int scNr = 0; scNr < scales.size(); scNr++) {
		vvImgDetect[scNr].resize(crForest->GetNumLabels());

		for (unsigned int lNr = 0; lNr < crForest->GetNumLabels(); ++lNr) {
			if ( (this_class >= 0 ) && (this_class != lNr) )
				continue;

			vvImgDetect[scNr][lNr] = Mat::zeros(Size(vImgAssign[scNr][0].cols * 2.0f + 0.5, vImgAssign[scNr][0].rows * 2.0f + 0.5), CV_32FC1);
		}

		voteColor(vImgAssign[scNr], vvImgDetect[scNr], -1, -1, this_class, default_rect, prob_threshold, classProbs[scNr], nvotes);
	}


	// detecting the peaks in the voting space
	detectPeaks(candidates, vvImgDetect, scales, max_cands, kernel_width[0] , kernel_width[2], true, -1, this_class, threshold);
}



// **********************************    LEAF ASSIGNMENT      ***************************************************** //

// matching the image to the forest and store the leaf assignments in vImgAssing
void CRForestDetector::assignCluster(Mat &img, Mat &depth_img, vector<Mat> &vImgAssign, float scale) {
	// extract features
	vector<Mat> vImg;
	CRPatch::extractFeatureChannels(img, depth_img, vImg, scale);

	// reset output image to -1 which indicates not matched regions
	Scalar initialAssignValue(-1.0);
	for (int i = 0; i < vImgAssign.size(); ++i)
		vImgAssign[i] = initialAssignValue;

	uchar **ptFCh     = new uchar*[vImg.size()];
	uchar **ptFCh_row = new uchar*[vImg.size()];
	int stepVImg = vImg[0].step[0];

	// x,y top left; cx,cy center of patch
	int xoffset = width / 2;
	int yoffset = height / 2;
	int x, y, cx, cy;

	cy = yoffset;
	for (y = 0; y < img.rows - height; ++y, ++cy) {

		// Get start of row
		for (unsigned int c = 0; c < vImg.size(); ++c)
			ptFCh_row[c] = vImg[c].ptr<uchar>(y);

		cx = xoffset;
		for (x = 0; x < img.cols - width; ++x, ++cx) {
			vector<const LeafNode *> result;

			crForest->regression(result, ptFCh_row, stepVImg);

			for (unsigned int treeNr = 0; treeNr < result.size(); treeNr++) {
				vImgAssign[treeNr].ptr<float>(y + yoffset)[x + xoffset] = float(result[treeNr]->idL);
			}

			// increase pointer - x
			for (unsigned int c = 0; c < vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for (unsigned int c = 0; c < vImg.size(); ++c)
			++ptFCh[c];

	} // end for y
}

// Multi-scale cluster assignment into vvImgAssign.
void CRForestDetector::fullAssignCluster(Mat &img, Mat &depth_img, vector<vector<Mat> > &vvImgAssign, vector<float> &scales) {

	vvImgAssign.resize(scales.size());

	int ntrees = crForest->vTrees.size();
	Scalar vvImgAssignValue(-1.0);

	for (unsigned int scaleNr = 0; scaleNr < scales.size(); scaleNr++) {
		vvImgAssign[scaleNr].resize(ntrees);
		// rescaling the image to scales[scaleNr]
		Mat cLevel, dcLevel;
		resize( img, cLevel, Size(img.cols * scales[scaleNr] + 0.5, img.rows * scales[scaleNr] + 0.5), 0, 0, CV_INTER_LINEAR );
		resize( depth_img, dcLevel, Size(depth_img.cols * scales[scaleNr] + 0.5, depth_img.rows * scales[scaleNr] + 0.5), 0, 0, CV_INTER_NN );

		//looping over the trees in the forest
		for (int treeNr = 0; treeNr < ntrees; treeNr++) {
			vvImgAssign[scaleNr][treeNr] = Mat(Size(img.cols * scales[scaleNr] + 0.5, img.rows * scales[scaleNr] + 0.5), CV_32FC1);
			vvImgAssign[scaleNr][treeNr] = vvImgAssignValue;
		}

		assignCluster(cLevel, dcLevel, vvImgAssign[scaleNr], scales[scaleNr]);
	}
}


// ************************************** CLASS CONFIDENCES ****************************************** //

// Getting the per class confidences TODO: this has to become scalable
void CRForestDetector::getClassConfidence(vector<vector<Mat> > &vImgAssign, vector<vector<Mat> > &classConfidence) {
	int nlabels = crForest->GetNumLabels();
	// allocating space for the classConfidence
	classConfidence.resize(vImgAssign.size());
	for (unsigned int i = 0; i < vImgAssign.size(); i++) {
		classConfidence[i].resize(nlabels);
		for (unsigned int j = 0; j < nlabels; j++) {
			classConfidence[i][j] = Mat::zeros(Size(vImgAssign[i][0].cols, vImgAssign[i][0].rows), CV_32FC1);
		}
	}

	// looping over the scales
	for (unsigned int scNr = 0; scNr < vImgAssign.size(); ++scNr) {

		int w = vImgAssign[scNr][0].cols;
		int h = vImgAssign[scNr][0].rows;

		// function variables
		int outer_window = 8; // TODO: this parameter shall move to the inputs.
		float inv_tree = 1.0f / vImgAssign[scNr].size();

		for (int trNr = 0; trNr < vImgAssign[scNr].size() ; trNr++) {
			// here make a temporary structure of all the probabilities and then smooth it with a kernel.
			vector<Mat> tmpClassProbs(nlabels);
			for (int cNr = 0; cNr < nlabels; ++cNr) {
				tmpClassProbs[cNr] = Mat::zeros(Size(vImgAssign[scNr][trNr].cols, vImgAssign[scNr][trNr].rows), CV_32FC1);
			}

			for (unsigned int y = 0; y < h ; ++y) {
				for (unsigned int x = 0; x < w; ++x) {
					int leaf_id = vImgAssign[scNr][trNr].ptr<float>(y)[x];
					if (leaf_id < 0)
						continue;

					LeafNode *tmp = crForest->vTrees[trNr]->getLeaf(leaf_id);

					for (int cNr = 0; cNr < nlabels; ++cNr) {
						tmpClassProbs[cNr].ptr<float>(y)[x] = tmp->vPrLabel[cNr] * inv_tree;
					}
				}
			}

			//SMOOTHING AND SCALING IF NECESSARY
			double scaleFactor = 1.0;
			if ( sample_points >= 0 ) {
				scaleFactor = 1.0 / (1.0 - sample_points);
			}
			// now values of the tmpClassProbs are set we can blur it to get the average
			for (int cNr = 0; cNr < nlabels; cNr++) {
				blur(tmpClassProbs[cNr], tmpClassProbs[cNr], Size(outer_window, outer_window));
				// to account for the subsampling
				resize(tmpClassProbs[cNr], tmpClassProbs[cNr], Size(), scaleFactor, scaleFactor);
			}

			for (int cNr = 0; cNr < nlabels; cNr++) {
				//  LOOPING OVER ALL PIXELS
				for (int y = 0; y < h; y++) {
					for (int x = 0 ; x < w; x++) {
						classConfidence[scNr][cNr].ptr<float>(y)[x] += tmpClassProbs[cNr].ptr<float>(y)[x];
					}
				}
			}
		}
	}
}
