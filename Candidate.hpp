#pragma once

#include "CRForest.hpp"
#include "utils.h"

#include <opencv2/core/core.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>



class Candidate {
public:
	// class constructors
	Candidate(CRForest::ConstPtr crForest, Mat &img, std::vector<float> candidate, int id, bool do_bpr = true);
	~Candidate() {
		clear();
	}


public:
	void clear();
	void getBBfromBpr(int thresh = 2, bool do_sym = false);
	void read(char *prefix) {}

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
	Mat backproj_mask; // backprojection image
};
