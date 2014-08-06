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

static std::vector< std::vector<IplImage*> > default_vvImg;
static std::vector< IplImage*> default_vImg;
static long int default_int = -1;

class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest* pRF, int w, int h, double s_points=-1.0 ,double s_forest=-1.0, bool bpr = true) : crForest(pRF), width(w), height(h), sample_points(s_points),do_bpr(bpr){
		crForest->GetClassID(Class_id);
	}

// Detection functions 
public: 
	void detectPyramidMR(std::vector<std::vector< IplImage* > >& vImgAssign, std::vector<std::vector<float> >& candidates, std::vector<float>& scales,std::vector<float>& ratios, std::vector<float> kernel_width , int max_cands=10,int this_class=-1,float threshold=0.0f,float prob_threshold=-1.0f, std::vector< std::vector<IplImage*> >& classProbs=default_vvImg, long int &nvotes = default_int);

	void voteForCandidate(std::vector< IplImage*>& imgAssign, Candidate& new_cand, int kernel_width, std::vector<float>& ratios, float max_width, float max_height  );

	void getClassConfidence(std::vector<std::vector<IplImage*> >& vImgAssign,std::vector<std::vector<IplImage*> >& classConfidence);

	void fullAssignCluster(IplImage *img, IplImage *depth_img, std::vector<std::vector< IplImage* > >& vvImgAssign, std::vector<float>& scales);

	void trainStat(IplImage* img, IplImage *depth_img, CvRect bbox, std::vector< std::vector<float> >& stat, float inv_set_size = 1.0f);

private:
	void assignCluster(IplImage* img, IplImage *depth_img, std::vector<IplImage*>& vImgAssign, float scale);

	void voteColor(std::vector<IplImage*>& imgAssign, std::vector<IplImage*>& imgDetect, float xShift=-1.0f, float yShift=-1.0f,int this_class=-1, CvRect* focus = NULL, float prob_threshold = -1, std::vector<IplImage*>& classProbs=default_vImg, long int &nvotes = default_int );

	void detectPeaks(std::vector<std::vector<float> >& candidates,std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float> scales, int max_cands, int kernel_width, float kernel_std, bool separate=true,float shift=-1.0f, int this_class = -1, float threshold = 0.0f);

/*************************************************************/

public:
	// Get/Set functions
	unsigned int GetNumLabels() const {return crForest->GetNumLabels();}
	void GetClassID(std::vector<std::vector<int> >& v_class_ids){
		crForest->GetClassID( v_class_ids);
	}
	bool GetHierarchy(std::vector<HNode>& hierarchy){// returns false if it could not load the hierarchy
		return crForest->GetHierarchy(hierarchy);
	}

	const CRForest* GetCRForest(){return crForest;}
//private variables
private: 
	const CRForest* crForest;
	std::vector<std::vector<int> > Class_id;
	int width;
	int height;
	double sample_points;
	bool do_bpr;
};
