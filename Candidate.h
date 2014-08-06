//
// C++ Implementation: Candidate
//
// Description: this class holds an structure for keeping all the information necessary for a candidate object hypothesis
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
//
//
//

#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <stdio.h>
#include <string>
#include "CRForest.h"
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include "utils.h"



/************************************/
/****     CLASS Candidate        ****/
/************************************/
// 
class Candidate{
public: 
    // class constructors
    Candidate(const CRForest* crForest, IplImage* img, std::vector<float> candidate, int id, bool do_bpr=true);
    ~Candidate(){clear();}
    

public:
	void clear();

	void getBBfromBpr(int thresh=2, bool do_sym=false);
	
	void print(char* prefix){
		// saving the backprojection mask
		if(bpr) save_bp( prefix);
	}

void read(char* prefix){}

private: 
	void save_bp(char* prefix);

public:
	float weight;
	float x; // width in this case
	float y; // height in this case
	float scale;
	int c;//class
	int r;//ratio
	int n_trees;
	int n_classes;
	int id;// candidate ID
	bool bpr; // if the backprojection is held
	
public:
	std::vector<float> bb; // bounding box structure
	IplImage* backproj_mask; // backprojection image
};
