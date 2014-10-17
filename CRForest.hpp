/**
 *  Training Procedures (training_mode):
 *      0: the training mode=0 does the InfGain over all classes
 *      1: transforms all the positive class_ids into different labels and does multi-class training with InfGain/nlabels + InfGainBG
 *      3: also transforms all the positive class ids into one label and does all-against-background training with InfGainBG
**/
#pragma once

#include "CRTree.hpp"

#define timer fubar
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>

#include <memory>
#include <vector>


using namespace std;


class CRForest {
public:
	typedef shared_ptr<CRForest> Ptr;
	typedef shared_ptr<CRForest const> ConstPtr;

	/* Trees */
	vector<CRTree::Ptr> vTrees_;
	/* training labels to use for detection */
	vector<int>  use_labels_;
	/* decide what kind of training procedures to take */
	int training_mode;

	CRForest(int num_trees = 0);
	~CRForest();
	void GetClassID(vector<vector<int> > &id) const;
	unsigned int GetDepth() const;
	bool GetHierarchy(vector<HNode> &hierarchy) const;
	unsigned int GetNumLabels() const;
	void SetTrees(int num_trees);
	int GetSize() const;

	void SetTrainingLabelsForDetection(vector<int> &class_selector);
	void GetTrainingLabelsForDetection(vector<int> &class_selector);
	// Regression
	void regression(vector<const LeafNode *> &result, vector<Mat> &vImg, int x, int y) const;
	// Training
	void trainForest(int min_s, int max_d, CvRNG *pRNG, const CRPatch &TrData, int samples, vector<int> &id, float scale_tree = 1.0f);
	// IO functions
	void saveForest(const char *filename, unsigned int offset = 0);
	bool loadForest(const char *filename, unsigned int offset = 0);
	void loadHierarchy(const char *hierarchy, unsigned int offset = 0);
};