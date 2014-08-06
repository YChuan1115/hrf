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
void CRForestDetector::voteColor(vector<IplImage*>& vImgAssign, std::vector<IplImage*>& vImgDetect, float xShift, float yShift,int this_class, CvRect* focus, float prob_threshold, std::vector<IplImage*>& classProbs, long int &nvotes){
// vImgDetect are all initialized before

	if (vImgAssign.size()<1)
		return;
	bool do_count =true;
	if (nvotes < 0)
		do_count = false;

 	// compensate for the shift based on the 	
	if (xShift < 0)
		 xShift = vImgAssign[0]->width * 0.50;
	if (yShift <0)
		 yShift = vImgAssign[0]->height * 0.50;


	int stepDet;
  	float** ptDet = new float*[vImgDetect.size()];
  	for(unsigned int cNr=0; cNr<vImgDetect.size(); ++cNr) {
		if ((this_class < 0) || (this_class == cNr)){
      			cvGetRawData( vImgDetect[cNr], (uchar**)&(ptDet[cNr]), &stepDet);
		}
  	}
  	stepDet /= sizeof(ptDet[0][0]);


	// the probabilites are for the 
	int stepProb;
  	float** ptProb = new float*[classProbs.size()];// be carefull of the image type
  	for(unsigned int i=0; i< classProbs.size(); ++i) {
		if ((this_class < 0) || (this_class == i)){
      		cvGetRawData( classProbs[i], (uchar**)&(ptProb[i]), &stepProb);			
		}
  	}
  	stepProb /= sizeof(ptProb[0][0]);

	float ntrees = float(vImgAssign.size());
	for (unsigned int trNr=0; trNr < vImgAssign.size(); trNr++){
		int x_begin,x_end,y_begin,y_end;
	
		if (vImgAssign[trNr]->roi != NULL){
			x_begin = vImgAssign[trNr]->roi->xOffset;		y_begin = vImgAssign[trNr]->roi->yOffset;
			x_end = vImgAssign[trNr]->roi->width + x_begin;	y_end = vImgAssign[trNr]->roi->height + y_begin;
		}else{
			x_begin = 0; 						y_begin = 0;
			x_end = vImgAssign[trNr]->width;		y_end = vImgAssign[trNr]->height;
		}

		for (unsigned int cy = y_begin ; cy < y_end; cy++){
			float* ptr = (float*) (vImgAssign[trNr]->imageData + cy * vImgAssign[trNr]->widthStep);
			for (unsigned int cx = x_begin; cx < x_end; cx++){	
				// get the leaf_id
				if(ptr[cx] <0)
					continue;

				LeafNode* tmp = crForest->vTrees[trNr]->getLeaf(ptr[cx]);
	
				for (unsigned int lNr=0; lNr < vImgDetect.size(); lNr++){
			
					if ((this_class >= 0 ) && (this_class != lNr)) // the voting should be done on a single class only
						continue;

					bool condition;
					if (prob_threshold < 0)
						condition = (Class_id[trNr][lNr]>0 && tmp->vPrLabel[lNr]*Class_id[trNr].size()>1);
					else{
						condition = (Class_id[trNr][lNr]>0  &&  *(ptProb[lNr]+cx+cy*stepProb) > prob_threshold);
					}

					if (condition){
						
						float w = tmp->vPrLabel[lNr] / ntrees;
						// vote for all points stored in a leaf
						vector<float>::const_iterator itW = tmp->vCenterWeights[lNr].begin();
						for(vector<CvPoint>::const_iterator it = tmp->vCenter[lNr].begin() ; it!=tmp->vCenter[lNr].end(); ++it, itW++) 
						{	
							int x = int(float(cx) - float((*it).x) + 0.5 + xShift);
							int y = int(float(cy) - float((*it).y) + 0.5 + yShift);
							if (do_count)
								nvotes++;

							if (focus==NULL){
								if(y>=0 && y<vImgDetect[lNr]->height && x>=0 && x<vImgDetect[lNr]->width) {
									*(ptDet[lNr]+x+y*stepDet) += w*(*itW);
								}
							}else{
								if (isInsideRect(focus,x,y)){
									*(ptDet[lNr]+(x-focus->x)+(y-focus->y)*stepDet) += w*(*itW);
								}
							}
						}
					}
				}
			}
		}
	}

  	delete[] ptDet;
	delete[] ptProb;
}


// Gathering the information in the support of each candidate
void CRForestDetector::voteForCandidate(vector< IplImage*>& vImgAssign, Candidate& cand, int kernel_width, std::vector<float>& ratios, float max_width, float max_height ){
	
	if (vImgAssign.size()<1)
		return;

	double sample_votes = 0.998;// ignore the patches by this probability

	double value= 0.0;

	time_t t = time(NULL);
	int seed = (int)t;
	CvRNG pRNG(seed);
			

	// get cand_x and cand_y at this scale
	int cand_x = int(cand.x * cand.scale + 0.5); 
	int cand_y = int(cand.y * cand.scale + 0.5);

	int min_x,min_y,max_x,max_y;

	// initializing the box around the candidate center where the votes can come from 
	min_x = cand_x - int(max_width/2.0f + kernel_width + 0.5f);		min_y = cand_y - int(max_height/2.0f + kernel_width + 0.5f);
	min_x = std::max(min_x,0);  									min_y = std::max(min_y,0);

	max_x = cand_x + int(max_width/2.0f + 0.5f);					max_y = cand_y + int(max_height/2.0f + 0.5f);
	max_x = std::min(vImgAssign[0]->width,max_x);					max_y = std::min(vImgAssign[0]->height,max_y);

	int x_offset = cand_x - min_x;
	int y_offset = cand_y - min_y;

	CvRect cand_roi;
	cand_roi.x = min_x; 				cand_roi.y = min_y; 
	cand_roi.width = max_x-min_x+1;		cand_roi.height = max_y-min_y+1;
	
	for (unsigned int trNr =0 ; trNr < vImgAssign.size(); trNr++)
		cvSetImageROI(vImgAssign[trNr], cand_roi);


	// looping over all trees
	float ntrees = float(vImgAssign.size());
	for (int trNr=0; trNr < int(ntrees); trNr++){


		int x_begin,x_end,y_begin,y_end;
		if (vImgAssign[trNr]->roi != NULL){
			x_begin = vImgAssign[trNr]->roi->xOffset;		y_begin = vImgAssign[trNr]->roi->yOffset;
			x_end = vImgAssign[trNr]->roi->width + x_begin;	y_end = vImgAssign[trNr]->roi->height + y_begin;
		}else{
			x_begin = 0; 							y_begin = 0;
			x_end = vImgAssign[trNr]->width;		y_end = vImgAssign[trNr]->height;
		}
		// looping over all locations 
		for (unsigned int cy = y_begin ; cy < y_end; cy++){
			float* ptr = (float*) (vImgAssign[trNr]->imageData + cy * vImgAssign[trNr]->widthStep);
			for (unsigned int cx = x_begin; cx < x_end; cx++){	
				
				value = cvRandReal(&pRNG);
				
				if (value < sample_votes || ptr[cx] <0)
					continue;

				LeafNode* tmp = crForest->vTrees[trNr]->getLeaf(ptr[cx]);

				float w = tmp->vPrLabel[cand.c] / ntrees;
				if (w< 0.0e-7)
					continue;

				float w_element = 0.0f;
				int idNr=0;
				vector<float>::const_iterator itW = tmp->vCenterWeights[cand.c].begin();
				for(vector<CvPoint>::const_iterator it = tmp->vCenter[cand.c].begin() ; it!=tmp->vCenter[cand.c].end(); ++it, ++idNr,itW++) {	
					int x = int(float(cx) - float((*it).x) * ratios[cand.r] + 0.5);
					int y = int(float(cy) - float((*it).y) + 0.5);
					if (isInsideKernel2D(x, y, cand_x,cand_y , kernel_width)){
						w_element += w*(*itW);
					}
				}								
				
				if ( w_element > 0.0 && cand.bpr){
					// update the backprojection image
					w_element += cvGetReal2D(cand.backproj_mask,int(cy),int(cx));
					cvSetReal2D(cand.backproj_mask, int(cy), int(cx), w_element);
				}
			}
		}
	}

	// resetting the image ROIs
	for (unsigned int trNr =0 ; trNr < vImgAssign.size(); trNr++)
		cvResetImageROI(vImgAssign[trNr]);
}



/********************************** FULL object detection ************************************/

void CRForestDetector::detectPeaks(std::vector<std::vector<float> >& candidates, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float> scales, int max_cands, int kernel_width, float kernel_std, bool separate, float shift, int this_class, float threshold){

candidates.clear();
	
// this is just to access a non-empty detect image for getting sizes and so on
int default_class = 0;
if ((this_class >= 0) )
	default_class = this_class;

// smoothing the accumulator matrix
std::vector<std::vector<IplImage*> > smoothAcc;
smoothAcc.resize(scales.size());
for (unsigned int scNr=0; scNr < scales.size(); scNr++){
	int adapKwidth = int(kernel_width*scales[scNr]/2.0f)*2 + 1; 
	float adapKstd  = kernel_std*scales[scNr];

	smoothAcc[scNr].resize(imgDetect[scNr].size());
	for (unsigned int cNr=0; cNr < imgDetect[scNr].size(); cNr++){
		if ((this_class >= 0) && ( this_class !=cNr))
			continue;
		smoothAcc[scNr][cNr] = cvCloneImage(imgDetect[scNr][cNr]); 
		cvSmooth(smoothAcc[scNr][cNr],smoothAcc[scNr][cNr],CV_GAUSSIAN,adapKwidth,adapKwidth,adapKstd);
	}
}

// each candidate is a six element vector weight, x, y, scale, class, ratio
CvPoint max_loc_temp;
CvPoint min_loc_temp;
double min_val_temp = 0;
double max_val_temp = 0;

float xShift;
float yShift;

/***************** find the local maximum locations **********************/
int candNr = 0;
for (unsigned int count=0; candNr < max_cands ; count++){// count can go until infinity
	bool flag = false;
	vector<float> max_position(6,-1);// max_val, x, y, scNr, cNr, rNr
	// detect the maximum	
	for (unsigned int scNr=0; scNr < scales.size(); scNr++){
		if (shift <0.0f){
			xShift = imgDetect[scNr][default_class]->width * 0.25;
			yShift = imgDetect[scNr][default_class]->height * 0.25;
		}else{
			xShift = imgDetect[scNr][default_class]->width * shift;
			yShift = imgDetect[scNr][default_class]->height * shift;
		}	
		for (unsigned int cNr=0; cNr < imgDetect[scNr].size(); cNr++){	
			if ((this_class >= 0) && ( this_class !=cNr))
				continue;
	
			cvMinMaxLoc(smoothAcc[scNr][cNr], &min_val_temp, &max_val_temp, &min_loc_temp, &max_loc_temp, NULL);
			if ( (max_val_temp >= threshold) && (max_val_temp > max_position[0]) ){
				flag = true;
				max_position[0]  = max_val_temp;
				max_position[1] = float(-xShift + max_loc_temp.x)/scales[scNr];
				max_position[2] = float(-yShift + max_loc_temp.y)/scales[scNr];
				max_position[3] = scales[scNr];
				if (separate){
					max_position[4] = cNr;
				}else{
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
	for (unsigned int scNr=0; scNr < scales.size(); scNr++){	
		if (max_position[3] != scales[scNr])
			continue;
		if (shift <0.0f){
			 xShift = imgDetect[scNr][default_class]->width * 0.25;
			 yShift = imgDetect[scNr][default_class]->height * 0.25;
		}else{
			xShift = imgDetect[scNr][default_class]->width * shift;
			yShift = imgDetect[scNr][default_class]->height * shift;
		}
		// remove the region with the supporting kernel width
		int adapHalfKwidth = int(kernel_width*(scales[scNr]/max_position[3])) + 1;
		float adapKstd  = kernel_std*scales[scNr];
		int cx = int(max_position[1]*scales[scNr] + xShift);
		int cy = int(max_position[2]*scales[scNr] + yShift);
		int x = std::max(0,cx-adapHalfKwidth);
		int y =std::max(0,cy-adapHalfKwidth);
		int rwidth = std::max(1,std::min(cx + adapHalfKwidth, smoothAcc[scNr][default_class]->width-1) - x + 1);
		int rheight = std::max(1,std::min(cy + adapHalfKwidth, smoothAcc[scNr][default_class]->height-1)  - y + 1);

		for (unsigned int cNr=0; cNr < imgDetect[scNr].size(); cNr++){
			if (max_position[4]>=0 && cNr!=max_position[4])
				continue;

			cvSetImageROI(smoothAcc[scNr][cNr],cvRect(x,y,rwidth,rheight));
			cvSetZero(smoothAcc[scNr][cNr]);
			cvResetImageROI(smoothAcc[scNr][cNr]);
		}
	}
}

//////////////     Releasing smoothed detection pyramid /////////////////////
for (unsigned int scNr=0; scNr < scales.size(); scNr++){
	for (unsigned int cNr=0; cNr < imgDetect[scNr].size(); cNr++){
		if ((this_class >= 0) && ( this_class !=cNr))
			continue;
			cvReleaseImage(&smoothAcc[scNr][cNr]);
	}
}

}

void CRForestDetector::detectPyramidMR(vector<vector< IplImage* > >& vImgAssign, vector<vector<float> >& candidates, std::vector<float>& scales,std::vector<float>& ratios, std::vector<float> kernel_width , int max_cands, int this_class,float threshold, float prob_threshold, std::vector< std::vector<IplImage*> >& classProbs, long int &nvotes){


	std::vector<std::vector<IplImage*> > vvImgDetect;
	vvImgDetect.resize(vImgAssign.size());
	
	// accumulating votes for all scales and classes
	for (unsigned int scNr=0; scNr < scales.size(); scNr++){
		vvImgDetect[scNr].resize(crForest->GetNumLabels());
		for (unsigned int lNr=0; lNr < crForest->GetNumLabels(); lNr++){
			if ( (this_class >= 0 ) && (this_class != lNr) )
				continue;

			vvImgDetect[scNr][lNr] = cvCreateImage( cvSize(int(vImgAssign[scNr][0]->width*2.0f + 0.5),int(vImgAssign[scNr][0]->height*2.0f + 0.5)) , IPL_DEPTH_32F , 1);
			cvSetZero(vvImgDetect[scNr][lNr]);
		}

		voteColor(vImgAssign[scNr],vvImgDetect[scNr],-1,-1,this_class,NULL,prob_threshold,classProbs[scNr],nvotes);
	}



	if (false) { /* debug view of voting space */
		IplImage* voting_space_resized = cvCreateImage(cvSize(480, 360), IPL_DEPTH_32F , 1); 
		char k;

		for (unsigned int scNr=0; scNr < vvImgDetect.size(); scNr++){
			stringstream ss;
			ss << "scale " << scNr;
			cvNamedWindow(ss.str().c_str(),1);
		}

		for (unsigned int scNr=0; scNr < vvImgDetect.size(); scNr++){
			for (unsigned int lNr=0; lNr < vvImgDetect[scNr].size(); lNr++){
				if ( (this_class >= 0 ) && (this_class != lNr) )
					continue;

				double min_val, max_val;
				cvResize(vvImgDetect[scNr][lNr], voting_space_resized);
				cvMinMaxLoc(voting_space_resized, &min_val, &max_val);
				std::cout << std::setprecision(3) << "\tscale " << scNr << ": [" << min_val << "," << max_val << "]" << std::endl;
				stringstream ss;
				ss << "scale " << scNr;
				cvShowImage(ss.str().c_str(), voting_space_resized);
			}
		}


		k = cvWaitKey();
		for (unsigned int scNr=0; scNr < vvImgDetect.size(); scNr++){
			stringstream ss;
			ss << "scale " << scNr;
			cvDestroyWindow(ss.str().c_str());
		}
	}




	// detecting the peaks in the voting space
	detectPeaks(candidates,vvImgDetect,scales,max_cands,kernel_width[0] , kernel_width[2],true,-1,this_class,threshold);

	// releasing the vImgDetect
	for (unsigned int scNr=0; scNr < scales.size(); scNr++){
		for (unsigned int lNr=0; lNr < crForest->GetNumLabels(); lNr++){
			if ( (this_class >= 0 ) && (this_class != lNr) )
				continue;
		
			cvReleaseImage(&vvImgDetect[scNr][lNr]);
		}		
	}

}



// **********************************    LEAF ASSIGNMENT      ***************************************************** //

// matching the image to the forest and store the leaf assignments in vImgAssing
void CRForestDetector::assignCluster(IplImage* img, IplImage *depth_img, vector<IplImage*>& vImgAssign, float scale){
// extract features
	vector<IplImage*> vImg;
	CRPatch::extractFeatureChannels(img, depth_img, vImg, scale);

	// reset output image to -1 which indicates not matched regions
	CvScalar scalar = {-1};
	for(int i=0; i<(int)vImgAssign.size(); ++i)    		
       cvSet( vImgAssign[i] , scalar);
	
	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh     = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for(unsigned int c=0; c<vImg.size(); ++c) {
		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	// get pointer to output image
	int stepDet;
  	float** ptDet = new float*[vImgAssign.size()];
  	for (unsigned int i=0; i<vImgAssign.size(); ++i) {
      			cvGetRawData( vImgAssign[i], (uchar**)&(ptDet[i]), &stepDet);
  	}
  	stepDet /= sizeof(ptDet[0][0]);


	int xoffset = width/2;
	int yoffset = height/2;

	double value= 0.0;

	time_t t = time(NULL);
	int seed = (int)t;
	CvRNG pRNG(seed);
	value = cvRandReal(&pRNG);

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset; 
	for(y=0; y < img->height - height; ++y, ++cy) {
		// Get start of row
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset; 
		
		for(x=0; x < img->width - width; ++x, ++cx) {
			vector<const LeafNode*> result;
			value = cvRandReal(&pRNG);
	
			bool do_regression = true;
			if (sample_points > 0 && value < sample_points)
					do_regression = false;
			
			if (do_regression){
				crForest->regression(result, ptFCh_row, stepImg);

				for (unsigned int treeNr=0; treeNr < result.size(); treeNr++){
					*(ptDet[treeNr]+(x+xoffset)+(y+yoffset)*stepDet) = float(result[treeNr]->idL);
				}
			}
						// increase pointer - x
			for(unsigned int c=0; c<vImg.size(); ++c)
				++ptFCh_row[c];
			
		} // end for x

		// increase pointer - y
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 	

	// release feature channels
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);

	delete[] ptFCh;
  	delete[] ptFCh_row;
  	delete[] ptDet;	

}

// Multi-scale cluster assignment into vvImgAssign. 
void CRForestDetector::fullAssignCluster(IplImage *img, IplImage *depth_img, vector<vector< IplImage* > >& vvImgAssign, vector<float>& scales){
	
	vvImgAssign.resize(scales.size());
	
	int ntrees = crForest->vTrees.size();
	CvScalar s = cvScalar(-1.0);

	for (unsigned int scaleNr =0; scaleNr < scales.size(); scaleNr++){
		vvImgAssign[scaleNr].resize(ntrees);
		// rescaling the image to scales[scaleNr]
		IplImage* cLevel = cvCreateImage(cvSize(int(img->width*scales[scaleNr]+0.5),int(img->height*scales[scaleNr]+0.5)), IPL_DEPTH_8U , 3);
		IplImage* dcLevel = cvCreateImage(cvSize(int(depth_img->width*scales[scaleNr]+0.5),int(depth_img->height*scales[scaleNr]+0.5)), IPL_DEPTH_16U , 1);
		cvResize( img, cLevel, CV_INTER_LINEAR );
		cvResize( depth_img, dcLevel, CV_INTER_NN );

		//looping over the trees in the forest
		for (int treeNr=0; treeNr < ntrees; treeNr++){
		    vvImgAssign[scaleNr][treeNr] =cvCreateImage(cvSize(int(img->width*scales[scaleNr]+0.5),int(img->height*scales[scaleNr]+0.5)), IPL_DEPTH_32F , 1);
			cvSet(vvImgAssign[scaleNr][treeNr],s);
		}

		assignCluster(cLevel, dcLevel, vvImgAssign[scaleNr], scales[scaleNr]);

		// clearing the memory 
		cvReleaseImage(&cLevel);
		cvReleaseImage(&dcLevel);
	}
}


// ************************************** CLASS CONFIDENCES ****************************************** //

// Getting the per class confidences TODO: this has to become scalable
void CRForestDetector::getClassConfidence(std::vector<std::vector<IplImage*> >& vImgAssign,std::vector<std::vector<IplImage*> >& classConfidence){
	int nlabels = crForest->GetNumLabels();	
	// allocating space for the classConfidence
	classConfidence.resize(vImgAssign.size());
	for (unsigned int i=0; i < vImgAssign.size(); i++){	
		classConfidence[i].resize(nlabels);
		for (unsigned int j=0; j < nlabels; j++){
			classConfidence[i][j] = cvCreateImage( cvSize(vImgAssign[i][0]->width,vImgAssign[i][0]->height), IPL_DEPTH_32F, 1 );
		}
	}

	// looping over the scales
	for (unsigned int scNr=0; scNr < vImgAssign.size(); scNr++){
			
		int h = vImgAssign[scNr][0]->height;
		int w = vImgAssign[scNr][0]->width;
		float ntrees = float(vImgAssign[scNr].size());

		// function variables
		int outer_window = 8; // TODO: this parameter shall move to the inputs. 
		float inv_tree = 1.0f/ntrees;

		for (int trNr=0; trNr < vImgAssign[scNr].size() ; trNr++){
			// first the assignment pointers
			int stepAss;
			float* ptAss;
			cvGetRawData(vImgAssign[scNr][trNr], (uchar**)&(ptAss),&stepAss); 
			stepAss /= sizeof(ptAss[0]);

			// here make a temporary structure of all the probabilities and then smooth it with a kernel.
			std::vector<IplImage*> tmpClassProbs(nlabels);
			int stepTmpProb;
			float** ptTmpProb = new float*[nlabels];
			for (int cNr=0; cNr < nlabels; cNr++){
				tmpClassProbs[cNr] = cvCreateImage( cvSize(vImgAssign[scNr][trNr]->width,vImgAssign[scNr][trNr]->height), IPL_DEPTH_32F, 1 );
				cvSetZero(tmpClassProbs[cNr]);
				cvGetRawData(tmpClassProbs[cNr],(uchar**)&(ptTmpProb[cNr]),&stepTmpProb);				
			}
			stepTmpProb /= sizeof(ptTmpProb[0][0]);

			for (unsigned int y =0; y < h ; y++){
				for (unsigned int x=0; x < w; x++){
					int leaf_id = *(ptAss+x+y*stepAss);
					LeafNode* tmp = crForest->vTrees[trNr]->getLeaf(leaf_id);
					for (int cNr=0; cNr < nlabels; cNr++){
						if (leaf_id >=0)
							*(ptTmpProb[cNr]+x+y*stepTmpProb) = tmp->vPrLabel[cNr]*inv_tree;
					}
				}
			}


			//SMOOTHING AND SCALING IF NECESSARY
			double scaleFactor = 1.0;
			if ( sample_points >= 0 ) {
				scaleFactor = 1.0/(1.0-sample_points);
			}
			// now values of the tmpClassProbs are set we can blur it to get the average
			for (int cNr=0; cNr<nlabels; cNr++){
				cvSmooth(tmpClassProbs[cNr],tmpClassProbs[cNr],CV_BLUR,outer_window,outer_window);
				// to account for the subsampling
				cvScale(tmpClassProbs[cNr],tmpClassProbs[cNr],scaleFactor);
			}

			for (int cNr=0; cNr < nlabels; cNr++){
				// get the class Confidence pointers
				int stepConf;
				float*ptConf ;
				cvGetRawData(classConfidence[scNr][cNr],(uchar**)&(ptConf),&stepConf);
				stepConf /= sizeof(ptConf[0]);
				//  LOOPING OVER ALL PIXELS
				for (int y = 0; y < h; y++){
					for (int x=0 ; x < w; x++){
						*(ptConf+x+y*stepConf) +=  *(ptTmpProb[cNr]+x+y*stepTmpProb);
					}
				}
			}

			// clearing memory of pointers
			delete[] ptTmpProb;
			// and the temporary images
			for (int cNr=0; cNr < nlabels; cNr++)
				cvReleaseImage(&tmpClassProbs[cNr]);
			
		}
	}
}


/***********************************************************************/

void CRForestDetector::trainStat(IplImage* img, IplImage *depth_img, CvRect bbox, std::vector< std::vector<float> >& stat, float inv_set_size){

	// extract features
	vector<IplImage*> vImg;
	CRPatch::extractFeatureChannels(img, depth_img, vImg, 1.0);

	std::vector<float> tree_scales;
	tree_scales.resize(crForest->vTrees.size());
	for (unsigned int i=0; i < tree_scales.size(); i++){
	 	tree_scales[i]= crForest->vTrees[i]->GetScale();
	}
	
	float inv_tree_scale = 1.0f;

	// get pointers to feature channels                
	int stepImg;
	uchar** ptFCh     = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for(unsigned int c=0; c<vImg.size(); ++c) {
		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);


	int xoffset = width/2;
	int yoffset = height/2;

	double value= 0.0;

	time_t t = time(NULL);
	int seed = (int)t;
	CvRNG pRNG(seed);
	
	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;


	float inv_num_feats =  bbox.height * bbox.width;
	inv_num_feats = 1.0f/inv_num_feats;

	for(y=0; y < img->height - height; ++y, ++cy) {
		if ((cy < bbox.y) || ( cy > bbox.y + bbox.height))
			continue;

		// Get start of row
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset; 
		
		for(x=0; x < img->width - width; ++x, ++cx) {
					
			if ((cx < bbox.x) || (cx > bbox.x + bbox.width))
				continue;
				
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row, stepImg);
				
			for(unsigned int tr=0;tr<result.size();++tr) {
				float w =  inv_num_feats * inv_set_size;
				stat[tr][result[tr]->idL] += w;
			}
			
			// increase pointer - x
			for(unsigned int c=0; c<vImg.size(); ++c)
				++ptFCh_row[c];
		} // end for x

		// increase pointer - y
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 	

	// release feature channels
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
	

	delete[] ptFCh;
  	delete[] ptFCh_row;
}