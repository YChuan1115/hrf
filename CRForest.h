#pragma once

#include "CRTree.h"

#define timer fubar
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>

#include <vector>


using namespace std;


class CRForest {
public:
	// Constructors
	CRForest(int trees = 0, bool doSkip=true) {
		vTrees.resize(trees);
		do_skip = doSkip;
		
	}
	~CRForest() {
		for(vector<CRTree*>::iterator it = vTrees.begin(); it != vTrees.end(); ++it) delete *it;
		vTrees.clear();
	}

	// Set/Get functions
	void SetTrees(int n) {vTrees.resize(n);}
	int GetSize() const {return vTrees.size();}
	unsigned int GetDepth() const {return vTrees[0]->GetDepth();}
	unsigned int GetNumLabels() const {return vTrees[0]->GetNumLabels();}
	void GetClassID(vector<vector<int> >& id) const;
	bool GetHierarchy(vector<HNode>& hierarchy) const{
		return vTrees[0]->GetHierarchy(hierarchy);
	}
	void SetTrainingLabelsForDetection(vector<int>& class_selector);
	void GetTrainingLabelsForDetection(vector<int>& class_selector);

	// Regression 
	void regression(vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const;

	// Training
	void trainForest(int min_s, int max_d, CvRNG* pRNG, const CRPatch& TrData, int samples, vector<int>& id, float scale_tree = 1.0f);

	// IO functions
	void saveForest(const char* filename, unsigned int offset = 0);
	bool loadForest(const char* filename, unsigned int offset = 0);
	void loadHierarchy(const char* hierarchy, unsigned int offset=0);

	// Trees
	vector<CRTree*> vTrees;

	// training labels to use for detection
	vector<int>  use_labels;

   // skipping training 
	bool do_skip;

	// decide what kind of training procedures to take
	int training_mode;// the normal information gain
	// the training mode=0 does the InfGain over all classes
	// the mode 1 transforms all the positive class_ids into different labels and does multi-class training with InfGain/nlabels + InfGainBG
	// the mode 3 also transforms all the positive class ids into one label and does all-against-background training with InfGainBG
};

// Matching 
inline void CRForest::regression(vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const {
	// if scale_tree == -1.0f -> process all the trees
	result.resize( vTrees.size() );
	for(int i=0; i<(int)vTrees.size(); ++i) {
		result[i] = vTrees[i]->regression(ptFCh, stepImg);
	}
}

//Training
inline void CRForest::trainForest(int min_s, int max_d, CvRNG* pRNG, const CRPatch& TrData, int samples, vector<int>& id, float scale_tree) {
	cout << "start training ..." << endl;
	boost::progress_display show_progress( vTrees.size() );
	
	for(int i=0; i < (int)vTrees.size(); ++i) {
		vTrees[i] = new CRTree(min_s, max_d, TrData.vLPatches.size(), pRNG);
		vTrees[i]->setClassId(id);
		vTrees[i]->SetScale(scale_tree);
		vTrees[i]->setTrainingMode(training_mode);
		vTrees[i]->growTree(TrData, samples);
		++show_progress;
	}
}

// IO Functions
inline void CRForest::saveForest(const char* filename, unsigned int offset) {
	char buffer[200];
	for(unsigned int i=0; i<vTrees.size(); ++i) {
		sprintf_s(buffer,"%s%03d.txt",filename,i+offset);
		vTrees[i]->saveTree(buffer);
	}
}

inline bool CRForest::loadForest(const char* filename, unsigned int offset) {
	char buffer[200];
	int cccc = 0;
	bool success=true;
	for(unsigned int i=offset; i <vTrees.size() + offset; ++i,++cccc) {		
		sprintf_s(buffer,"%s%03d.txt",filename,i);
		bool s;
		vTrees[cccc] = new CRTree(buffer,s);
		success *=s;		
		//vTrees[cccc]->saveTree(buffer);
	}
	return success;
}

inline void CRForest::loadHierarchy(const char* hierarchy, unsigned int offset){
	char buffer[400];
	int cccc =0;
	for (unsigned int i=offset; i < vTrees.size() + offset; ++i,++cccc){
		if(!(vTrees[cccc]->loadHierarchy(hierarchy))){
			cerr<< "failed to load the hierarchy: " << hierarchy << endl;
		}else{
			cout<< "loaded the hierarchy: " << hierarchy << endl;
		}
	}
}


// Get/Set functions 
inline void CRForest::GetClassID(vector<vector<int> >& id) const {
  id.resize(vTrees.size());
  for(unsigned int i=0; i<vTrees.size(); ++i) {
    vTrees[i]->getClassId(id[i]);
  }  
}

inline void CRForest::SetTrainingLabelsForDetection(vector<int>& class_selector){
	int nlabels = GetNumLabels();
	if (class_selector.size()==1 && class_selector[0]==-1){
		use_labels.resize(nlabels);
		cout<< nlabels << " labels used for detections:" << endl;
		for (unsigned int i=0; i < nlabels; i++){
			use_labels[i] = 1;
	//		cout << " label " << i << " : " << use_labels[i] << endl; 
		}
	}else{
		if ((unsigned int)(nlabels)!= class_selector.size()){
			cerr<< "nlabels: " << nlabels << " class_selector.size(): " << class_selector.size() << endl; 
			cerr<< "CRForest.h: the number of labels does not match the number of elements in the class_selector" << endl;
			return;
		}
		use_labels.resize(class_selector.size());
		for (unsigned int i=0; i< class_selector.size(); i++){	
			use_labels[i] = class_selector[i];
		}
	}
		
}

inline void CRForest::GetTrainingLabelsForDetection(vector<int>& class_selector){ 
	class_selector.resize(use_labels.size());
	for (unsigned int i=0; i< use_labels.size(); i++)
		class_selector[i] = use_labels[i];
}