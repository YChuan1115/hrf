/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
//
// Modified: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
*/

#pragma once

#define sprintf_s sprintf 

#include "CRPatch.h"
#include <iostream>
#include <fstream>

// Auxilary structure
struct IntIndex {
	int val;
	unsigned int index;
	bool operator<(const IntIndex& a) const { return val<a.val; }
};

// Structure for the leafs
struct LeafNode {
	// Constructors
	LeafNode() {}

	// IO functions
	const void show(int delay, int width, int height, int* class_id); 
	const void print() const {
	  std::cout << "Leaf " << vCenter.size() << " ";
	  for(unsigned int c = 0; c<vCenter.size(); ++c) 
	    std::cout << vCenter[c].size() << " "  << vPrLabel[c] << " ";
	  std::cout << std::endl;
	}
	float cL; // what proportion of the entries at this leaf is from foreground
	int idL; // leaf id 
	float fL; //occurrence frequency 
	float eL;// emprical probability of when a patch is matched to this cluster, it belons to fg
	std::vector<int> nOcc;
	std::vector<float> vLabelDistrib;
	// Probability of foreground
	std::vector<float> vPrLabel;
	// Vectors from object center to training patches
	std::vector<std::vector<Point> > vCenter;
	std::vector<std::vector<float> > vCenterWeights;
	std::vector<std::vector<int> > vID;
};

// Structure for internal Nodes
struct InternalNode {
	// Constructors
	InternalNode() {}

	// Copy Constructor
	InternalNode(const InternalNode& arg){
		parent = arg.parent;
		leftChild = arg.leftChild;
		rightChild = arg.rightChild;
		idN = arg.idN;
		depth = arg.depth;
		data.resize(arg.data.size());
		for (unsigned int dNr=0; dNr < arg.data.size(); dNr++)
			data[dNr] = arg.data[dNr];
		isLeaf = arg.isLeaf;
	}

	// relative node Ids
	int parent; // parent id, if this node is root, the parent will be -1
	int leftChild; // stores the left child id, if leaf stores the leaf id
	int rightChild;// strores the right child id, if leaf is set to -1

	//internal data
	int idN;//node id	

	int depth;
	
	//	the data inside each not
	std::vector<int> data;// x1 y1 x2 y2 channel threshold
	bool isLeaf;// if leaf is set to 1 otherwise to 0, the id of the leaf is stored at the left child

};

struct HNode {
	HNode() {}

	// explicit copy constructor
	HNode(const HNode& arg){
		id = arg.id;
		parent = arg.parent;
		leftChild = arg.leftChild;
		rightChild = arg.rightChild;
		subclasses = arg.subclasses;
		linkage = arg.linkage;
	}
	
	bool isLeaf(){
		return ((leftChild < 0) && (rightChild < 0));
	}
	
	int id;
	int parent;// stores the id of the parent node: if root -1
	int leftChild; // stores the id of the left child, if leaf -1
	int rightChild;// stores the id of the right child, if leaf -1
	float linkage;
	std::vector<int> subclasses; // stores the id of the subclasses which are under this node, 
};

class CRTree {
public:
	// Constructors
	CRTree(const char* filename, bool& success);
	CRTree(int min_s, int max_d, int l, CvRNG* pRNG) : min_samples(min_s), max_depth(max_d), num_leaf(0), num_nodes(1), num_labels(l), cvRNG(pRNG) {
		this->id = CRTree::treeCount++;

		nodes.resize(int(num_nodes));
		nodes[0].isLeaf = false;
		nodes[0].idN = 0; // the id is set to zero for the root
		nodes[0].leftChild = -1;	
		nodes[0].rightChild = -1;
		nodes[0].parent = -1;
		nodes[0].data.resize(6,0);
		nodes[0].depth = 0;
		
		//initializing the leafs
		leafs.resize(0);
		// class structure
		class_id = new int[num_labels];
	}
	~CRTree() {  delete[] class_id;}//clearLeaves(); clearNodes();

	// Set/Get functions
	unsigned int GetDepth() const {return max_depth;}
	unsigned int GetNumLabels() const {return num_labels;}
	void setClassId(std::vector<int>& id) {
	  for(unsigned int i=0;i<num_labels;++i) class_id[i] = id[i];
	}
	void getClassId(std::vector<int>& id) const {
	  id.resize(num_labels);
	  for(unsigned int i=0;i<num_labels;++i) id[i] = class_id[i];
	}
	float GetScale() {return scale;}
	void SetScale(const float tscale) {scale = tscale;}

	int getNumLeaf(){return num_leaf;}
	LeafNode* getLeaf(int leaf_id=0){return &leafs[leaf_id];}

	void setTrainingMode(int mode){training_mode = mode;}

	bool GetHierarchy(std::vector<HNode>& h){
	 	if ( (hierarchy.size()==0) ) { // check if the hierarchy is set at all(hierarchy == NULL) ||
			return false;
		}
		h = hierarchy;
		return true;	
	};

	// Regression
	const LeafNode* regression(uchar** ptFCh, int stepImg) const;

	// Training
	void growTree(const CRPatch& TrData, int samples);

	// IO functions
	bool saveTree(const char* filename) const;
	bool loadHierarchy(const char* filename);

private: 

	static unsigned int treeCount;
	unsigned int id;

	// Private functions for training
	void grow(const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector<int> >& TrainIDs, int node, unsigned int depth, int samples, std::vector<float>& vRatio);

	int getStatSet(const std::vector<std::vector<const PatchFeature*> >& TrainSet, int* stat);

	void makeLeaf(const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector< int> >& TrainIDs , std::vector<float>& vRatio, int node);

	bool optimizeTest(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, std::vector<std::vector<int> >& idA, std::vector<std::vector<int> >& idB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector<int> >& TrainIDs, int* test, unsigned int iter, unsigned int mode, const std::vector<float>& vRatio);



	void generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c);

	void evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet);

	void split(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, std::vector<std::vector<int > >& idA, std::vector<std::vector<int> >& idB , const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector<IntIndex> >& valSet, const std::vector<std::vector<int> >& TrainIDs ,int t);

	double measureSet(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB, unsigned int mode, const std::vector<float>& vRatio) {
	  if (mode==0) {
		if (training_mode==0){ // two class information gain
			return InfGain(SetA, SetB, vRatio); 
		}else if(training_mode ==1){// multiclass infGain with background
			return InfGainBG(SetA,SetB,vRatio) + InfGain(SetA,SetB,vRatio)/double(SetA.size());
		}else if(training_mode ==2){ // multiclass infGain without background
			return InfGain(SetA,SetB,vRatio);
		}else{
			std::cerr << " there is no method associated with the training mode: " << training_mode << std::endl;
			return -1;
		}
	  }else {
		if (training_mode==2 || training_mode ==0){
			return -distMean(SetA,SetB);
		}else{
			return -distMeanMC(SetA,SetB);
		}
	  }
	}

	double distMean(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB);
	double distMeanMC(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB);

	double InfGain(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<float>& vRatio);

	double InfGainBG(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<float>& vRatio);
	
	// Data structure

	// tree table
	// 2^(max_depth+1)-1 x 7 matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	//int* treetable;

	// stop growing when number of patches is less than min_samples
	unsigned int min_samples;

	// depth of the tree: 0-max_depth
	unsigned int max_depth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int num_nodes;

	// number of leafs
	unsigned int num_leaf;

	// number of labels
	unsigned int num_labels;

	// classes
	int* class_id;

	int training_mode;// 1 for multi-class detection

	// scale of the training data with respect to some reference
	float scale;
	//leafs as vector
	std::vector<LeafNode> leafs;

	// internalNodes as vector
	std::vector<InternalNode> nodes;// the first element of this is the root

	// hierarchy as vector
	std::vector<HNode> hierarchy;
	CvRNG *cvRNG;
};

inline const LeafNode* CRTree::regression(uchar** ptFCh, int stepImg) const {
	// pointer to the current node first set to the root node
	//InternalNode* pnode = &nodes[0];
	
	int node = 0;

	// Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
	while(!nodes[node].isLeaf) {
		// binary test 0 - left, 1 - right
		// Note that x, y are changed since the patches are given as matrix and not as image 
		// p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false
		
		// pointer to channel
		uchar* ptC = ptFCh[nodes[node].data[4]];
		// get pixel values 
		int p1 = ptC[nodes[node].data[0]+nodes[node].data[1]*stepImg];
		int p2 = ptC[nodes[node].data[2]+nodes[node].data[3]*stepImg];
		// test
		bool test = ( p1 - p2 ) >= nodes[node].data[5];

		// next node is at the left or the right child depending on test
		if (test)
			node = nodes[node].rightChild;
		else
			node = nodes[node].leftChild;
		
	}

	return &leafs[nodes[node].leftChild];
}

inline void CRTree::generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c) {
	test[0] = cvRandInt( cvRNG ) % max_w;
	test[1] = cvRandInt( cvRNG ) % max_h;
	test[2] = cvRandInt( cvRNG ) % max_w;
	test[3] = cvRandInt( cvRNG ) % max_h;
	test[4] = cvRandInt( cvRNG ) % max_c;
}

inline int CRTree::getStatSet(const std::vector<std::vector<const PatchFeature*> >& TrainSet, int* stat) {
  int count = 0;
  for(unsigned int l=0; l<TrainSet.size(); ++l) {
    if(TrainSet[l].size()>0) 
      stat[count++]=l;
  }
  return count;
}
