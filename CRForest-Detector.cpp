// Authors:  Nima Razavi, BIWI, ETH Zurich
//           Juergen Gall, BIWI, ETH Zurich
// Email:    nrazavi@vision.ee.ethz.ch
//           gall@vision.ee.ethz.ch
//
//


#include <fstream>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "CRForestDetector.h"



using namespace std;
using namespace cv;

#define PATH_SEP "/"


// Path to trees
string treepath;
// Path to the hierarchy structure
string hierarchy;
// variable enabling the hierarchical detection
bool do_hierarchy;
// the Alpha in []
float threshold_hierarchy;
// Number of trees
int ntrees;
//Tree depth
int treedepth;
// Number of classes
int nlabels;
// Patch width
int p_width;
// Patch height
int p_height;
// Path to images
string impath;
// File with names of images
string imfiles;
// Extract features
bool xtrFeature;
// Scales
vector<float> scales;
// Scales per class
vector<float> scales_per_class;
// Ratio
vector<float> ratios;
// Output path
string outpath;
// scale factor for output image (default: 128)
int out_scale;
// Path to training examples
string trainclasspath;
// File listing training examples from each class
string trainclassfiles;
// Path to recentering examples
string recenterpath;
// File with postive examples
string recenterfiles;
// Subset of positive images -1: all images
int subsamples_class;
// Subset of positive images -1: all images
int subsamples_class_neg;
// Sample patches from pos. examples
unsigned int samples_class;
// Class structure
vector<int> class_structure;
// scale of the tree
float scale_tree = -1.0f;
// offset for saving tree number
int off_tree;
// test image number to sprocess
int off_test;
// test class to process
int select_test_class;
// test set number
int select_test_set;
// number of test images to be processed
int test_num;
// running the detection
int do_detect = 1;
// The smoothing kernel parameters
float *kernel_width;
// threshold for the detection
float theta = 0.01f;
// number of candidates per class
int max_candidates = 10;
// maximum width and height of an object's bounding box
float max_width = 75.0f;
float max_height = 75.0f;
// sampling probability
double sample_points = -1.0;
// set this variable to enable skipping the already calculated detection
bool doSkip;
// backprojecting the bounding box
bool do_bpr;
// setting these variables to determine what classes to do detection/training and test with
vector<int> train_classes, detect_classes, emp_classes;
// type of training: allVSBG=0 multiClassTraining=1 MixedMultiClassBG=2
int training_mode;
bool DEBUG = false;



// load config file for dataset
void loadConfig(const char *filename, int mode) {
	char buffer[400];
	ifstream in(filename);

	if (in.is_open()) {

		// Path to trees
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		treepath = buffer;
		// Number of trees
		in.getline(buffer, 400);
		in >> ntrees;
		in.getline(buffer, 400);
		// Depth of tree
		in.getline(buffer, 400);
		in >> treedepth;
		in.getline(buffer, 400);
		// Patch width
		in.getline(buffer, 400);
		in >> p_width;
		in >> p_height;
		in.getline(buffer, 400);
		// File with names of images
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		imfiles = buffer;
		// Scales
		in.getline(buffer, 400);
		int size;
		in >> size;
		scales.resize(size);
		for (int i = 0; i < size; ++i)
			in >> scales[i];
		in.getline(buffer, 400);
		// Output path
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		outpath = buffer;
		// File with train examples
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		trainclassfiles = buffer;
		// Subset of train sequences -1: all sequences
		in.getline(buffer, 400);
		in >> subsamples_class;
		in.getline(buffer, 400);
		// Subset of neg. train sequences -1: all sequences
		in.getline(buffer, 400);
		in >> subsamples_class_neg;
		in.getline(buffer, 400);
		// Samples from sequences
		in.getline(buffer, 400);
		in >> samples_class;
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in >> size;
		kernel_width = new float[size];
		//cout << size <<endl;
		for (int i = 0; i < size; ++i)
			in >> kernel_width[i];//kernel_width; [3] = {20.0,20.0,0.4};
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in >> theta;
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in >> max_candidates;
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in >> max_width;
		in >> max_height;
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in >> doSkip;
		in >> do_bpr;
		// mode of training: allVSBG=0 multiClassTraining=1 MixedMultiClassBG=2
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in >> training_mode;
		// hierarchy structure
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		//std::cerr<< buffer << std::endl;
		hierarchy = buffer;
		// variable for doing the hierarchical detection or not
		in.getline(buffer, 400);
		in >> do_hierarchy;
		in >> threshold_hierarchy;
		//std::cerr<< do_hierarchy << std::endl;
		// scales per class
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		size = 0;
		in >> size;
		scales_per_class.resize(size);
		std::cout << " scales per class " ;
		for (int i = 0; i < size; ++i) {
			in >> scales_per_class[i];
			std::cout << " " << scales_per_class[i];
		}
		std::cout << std::endl;
	} else {
		cerr << "Config file not found " << filename << endl;
		exit(-1);
	}
	in.close();

	switch ( mode ) {
	case 0:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Training:         " << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Train pos:        " << trainclasspath << endl;
		cout << "                  " << trainclassfiles << endl;
		cout << "                  " << subsamples_class << " " << subsamples_class_neg << " " << samples_class << endl;
		cout << "Trees:            " << ntrees << " " << off_tree << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

	case 1:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Show:             " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

	default:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Detection:        " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Images:           " << impath << endl;
		cout << "                  " << imfiles << endl;
		cout << "Scales:           "; for (unsigned int i = 0; i < scales.size(); ++i) cout << scales[i] << " "; cout << endl;
		cout << "Ratios:           "; for (unsigned int i = 0; i < ratios.size(); ++i) cout << ratios[i] << " "; cout << endl;
		cout << "Extract Features: " << xtrFeature << endl;
		cout << "Output:           " << out_scale << " " << outpath << endl;
		cout << "Skipping:         " << doSkip << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;
	}

}


// load training image filenames
void loadTrainClassFile(string trainclass_files , std::vector<std::vector<string> > &vFilenames, std::vector<std::vector<Rect> > &vBBox, std::vector<std::vector<Point> > &vCenter, std::vector<string> &internal_files) {

	ifstream in_class(trainclass_files.c_str());

	if (in_class.is_open()) {

		in_class >> nlabels;
		vFilenames.resize(nlabels);
		vBBox.resize(nlabels);
		vCenter.resize(nlabels);
		class_structure.resize(nlabels);
		cout << "Classes: " << nlabels << endl;
		string labelfile;
		internal_files.resize(nlabels);

		for ( int l = 0; l < nlabels; ++l) {


			in_class >> class_structure[l];
			in_class >> labelfile;
			internal_files[l] = labelfile;

			ifstream in(labelfile.c_str());
			if (in.is_open()) {

				unsigned int size, dummy;
				in >> size; in >> dummy;

				cout << "Load Train Examples: " << l << " - " << size << endl;
				cout  << labelfile.c_str() << endl;

				vFilenames[l].resize(size);
				vCenter[l].resize(size);
				vBBox[l].resize(size);

				for (unsigned int i = 0; i < size; ++i) {
					// Read filename
					in >> vFilenames[l][i];
					//cout << vFilenames[l][i]<<endl;
					// Read bounding box
					in >> vBBox[l][i].x; in >> vBBox[l][i].y;
					in >> vBBox[l][i].width;
					vBBox[l][i].width -= vBBox[l][i].x;
					in >> vBBox[l][i].height;
					vBBox[l][i].height -= vBBox[l][i].y;

					if (vBBox[l][i].width < p_width || vBBox[l][i].height < p_height) {
						cout << "Width or height are too small" << endl;
						cout << "width: " << vBBox[l][i].width << "height: " << vBBox[l][i].height << endl;
						cout << vFilenames[l][i] << endl;
						exit(-1);
					}

					// Read center points
					in >> vCenter[l][i].x;
					in >> vCenter[l][i].y;
				}

			} else {
				cerr << "File not found " << labelfile.c_str() << endl;
				exit(-1);
			}

			in.close();

		}


	} else {
		cerr << "File not found " << trainclass_files.c_str() << endl;
		exit(-1);
	}

	in_class.close();

}



// Extract patches from training data
void extract_Patches(CRPatch &Train, CvRNG *pRNG) {

	vector<vector<string> > vFilenames;
	vector<vector<Rect> > vBBox;
	vector<vector<Point> > vCenter;
	vector<string> internal_files;

	// load positive file list
	loadTrainClassFile(trainclassfiles, vFilenames,  vBBox, vCenter, internal_files);
	Train.setClasses(nlabels);


	// for each class/label
	for (int l = 0; l < nlabels; ++l) {

		cout << "Label: " << l << " " << class_structure[l] << " ";

		int subsamples = 0;
		if (class_structure[l] == 0)
			subsamples = subsamples_class_neg;
		else
			subsamples = subsamples_class;

		// load postive images and extract patches
		for (int i = 0; i < (int)vFilenames[l].size(); ++i) {

			if (i % 50 == 0) cout << i << " " << flush;

			if (subsamples <= 0 || (int)vFilenames[l].size() <= subsamples || (cvRandReal(pRNG)*double(vFilenames[l].size()) < double(subsamples)) ) {

				// Load image
				Mat img, depth_img;

				string img_file_name = vFilenames[l][i];
				string img_depth_file_name = img_file_name;
				int start_pos = img_file_name.find("crop.png");
				if (start_pos != -1) {
					img_depth_file_name.replace(start_pos, 8, "depthcrop.png");
				} else {
					start_pos = img_file_name.find(".png");
					img_depth_file_name.replace(start_pos, 4, "_depth.png");
				}


				img = imread(trainclasspath + PATH_SEP + img_file_name, CV_LOAD_IMAGE_COLOR);
				// is going to be IPL_DEPTH_16U
				depth_img = imread(trainclasspath + PATH_SEP + img_depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);

				if (!img.data) {
					cout << "Could not load image file: " << trainclasspath + PATH_SEP + img_file_name << endl;
					exit(-1);
				} else if (!depth_img.data) {
					cout << "Could not load image file: " << trainclasspath + PATH_SEP + img_depth_file_name << endl;
					exit(-1);
				}

				// Extract positive training patches
				Train.extractPatches(img, depth_img, samples_class, l, i , vBBox[l][i], vCenter[l][i]);
			}
		}
		cout << endl;
	} cout << endl;
}



// Init and start training
void run_train() {

	// Init forest with number of trees
	CRForest crForest( ntrees , doSkip);

	if (doSkip && crForest.loadForest(treepath.c_str(), off_tree)) {
		return; // the forest is already trained
	}

	crForest.training_mode = training_mode;

	// Init random generator
	time_t t = time(NULL);
	int seed = (int)(t / double(off_tree + 1));//1407685013;

	CvRNG cvRNG(seed);

	cout << " seed " << seed << endl;

	// Create directory
	string tpath(treepath);
	tpath.erase(tpath.find_last_of(PATH_SEP));
	string execstr = "mkdir ";
	execstr += tpath;
	system( execstr.c_str() );

	// Init training data
	CRPatch Train(&cvRNG, p_width, p_height); //, 2);

	// Extract training patches
	extract_Patches(Train, &cvRNG);

	// depending on the training mode you should change the class_structure
	if (training_mode == 0) {
		std::cout << " the class labels have kept the way they are" << std::endl;
	} else {
		// only keep the label of the background class as 0 and the rest should just get labelled differntly
		std::cout << " the class labels have changed: the background class(with label 0) is kept and all other classes have assigned different labels according to their rank in the training file" << std::endl;
		// first check if there are only 0 and 1 in the class_structure
		bool binary = true;
		for (int i = 0; i < class_structure.size() ; i++) {
			if (class_structure[i] > 1 || class_structure[i] < 0)
				binary = false;
		}
		if (binary) {
			int count = 1;
			std::cout << " new class labels: " << std::endl;
			for (int i = 0; i < class_structure.size(); i++) {
				if (class_structure[i] != 0) {
					class_structure[i] = count;
					count++;
				}
				std::cout << " label: " << i << " " << class_structure[i] << std::endl;
			}
		} else {
			std::cout << "there are two classes only, training mode changed to 0" << std::endl;
			crForest.training_mode = 0;
		}
	}

	// Train forest
	crForest.trainForest(20, treedepth, &cvRNG, Train, 2000, class_structure, scale_tree);

	bool ignore_statistics = true;

	// initializing some statistics in to the leaf nodes
	for (unsigned int trNr = 0; trNr < crForest.vTrees.size(); ++trNr) {
		LeafNode *leaf = crForest.vTrees[trNr]->getLeaf();
		LeafNode *ptLN = &leaf[0];
		std::vector<int> class_ids;
		crForest.vTrees[trNr]->getClassId(class_ids);

		for (unsigned int lNr = 0 ; lNr < crForest.vTrees[trNr]->getNumLeaf(); lNr++, ++ptLN) {
			ptLN->eL = 0;
			ptLN->fL = 0;
			ptLN->vLabelDistrib.resize(class_ids.size(), 0);
		}
	}
	// Save forest
	crForest.saveForest(treepath.c_str(), off_tree);
}



// load testing image filenames
void loadTestClassFile(std::vector<std::vector<string> > &vFilenames) {

	ifstream in_class(imfiles.c_str());
	if (in_class.is_open()) {
		int n_test_classes;
		in_class >> n_test_classes;
		vFilenames.resize(n_test_classes);
		//test_classes.resize(n_test_classes);

		cout << "number Classes: " << vFilenames.size() << endl;
		string labelfile;
		for (int l = 0; l < n_test_classes; ++l) {
			in_class >> labelfile;
			ifstream in(labelfile.c_str());
			if (in.is_open()) {
				unsigned int size;
				in >> size;
				cout << "Load Test Examples: " << l << " - " << size << endl;
				vFilenames[l].resize(size);
				for (unsigned int i = 0; i < size; ++i) {
					// Read filename
					in >> vFilenames[l][i];
				}
			} else {
				cerr << "File not found " << labelfile.c_str() << endl;
				exit(-1);
			}
			in.close();
		}
	} else {
		cerr << "File not found " << imfiles.c_str() << endl;
		exit(-1);
	}
	in_class.close();
}



void detect(CRForestDetector &crDetect) {
	ratios.resize(1);
	ratios[0] = 1;

	// Load image names
	vector<vector<string> > vFilenames;
	loadTestClassFile(vFilenames);

	char buffer[3000];
	char buffer2[3000];
	char buffer3[3000];

	int file_test_num = 0;
	std::cout << "start detection ... " << std::endl;
	for (unsigned int tcNr = 0; tcNr < vFilenames.size(); tcNr++) {

		if ( select_test_set > 0 && int(tcNr) != select_test_set) {
			continue;
		}

		if (test_num < 0) { // number of test images to process
			file_test_num = vFilenames[tcNr].size() - off_test;
		} else {
			file_test_num = test_num;
		}

		// find the largest detetion scale for edge detection
		float max_scale = 1.0f;
		for (int s = 0; s < scales.size() ; s++) {
			if (max_scale < scales[s])
				max_scale = scales[s];
		}


		// Run detector for each image
		boost::progress_display show_progress( file_test_num - off_test );
		for (int i = off_test; i < off_test + file_test_num; ++i) {
			++show_progress;

			if (i >= vFilenames[tcNr].size())
				continue;

			// Creat directory for the result
			sprintf_s(buffer2, "%s/detect_o%d_n%d-%d-%d_cand_all", outpath.c_str(), off_tree, ntrees, tcNr, i);
			sprintf_s(buffer3, "mkdir %s", buffer2);
			system(buffer3);

			// check the files
			bool skipping = false;
			if (doSkip) {
				skipping = true;
				// first check if the candidates.txt file exists.
				char cand_dir[3000];
				sprintf_s(cand_dir, "%s", buffer2);
				char cand_filename[3000];
				sprintf_s(cand_filename, "%s/candidates.txt", cand_dir);

				FILE *file;
				if (file = fopen(cand_filename, "r")) {
					fclose(file);
					// read the candidates
					std::ifstream cand_file;
					cand_file.open(cand_filename);
					int cand_size = 0;
					cand_file >> cand_size;
					cand_file.close();

					char backpr_filename[3000];
					sprintf_s(backpr_filename, "%s/boundingboxes.txt", cand_dir);
					if (file = fopen(backpr_filename, "r")) {
						fclose(file);
						std::ifstream backpr_file;
						backpr_file.open(backpr_filename);
						int boxes_size = 0;
						backpr_file >> boxes_size;
						backpr_file.close();

						if (boxes_size != cand_size) {
							skipping = false;
						}
					} else {
						skipping = false;
					}

				} else {
					skipping = false;
				}
			}
			// the detections are already calculated
			if (skipping) {
				continue;
			}

			// FROM HERE THE DETECTION STARTS
			// Load image
			Mat img, depth_img;

			string img_file_name = vFilenames[tcNr][i];
			string img_depth_file_name = img_file_name;
			int start_pos = img_file_name.find("crop.png");
			if (start_pos != -1) {
				img_depth_file_name.replace(start_pos, 8, "depthcrop.png");
			} else {
				start_pos = img_file_name.find(".png");
				img_depth_file_name.replace(start_pos, 4, "_depth.png");
			}

			img = imread(impath + PATH_SEP +  img_file_name, CV_LOAD_IMAGE_COLOR);
			depth_img = imread(impath + PATH_SEP + img_depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);

			if (!img.data || !depth_img.data) {
				cout << "Could not load image file: " << (trainclasspath + PATH_SEP + img_file_name) << endl;
				exit(-1);
			}

			// preparation
			std::vector<HNode> h;
			std::vector<int> id2h;
			if (do_hierarchy) {
				if (!crDetect.GetHierarchy(h)) {
					std::cerr << "unable to load the hierarchy" << std::endl;
					return;
				}

				// make a vector getting the ids and returning the hierarchy node, all the ids should be bq to zero
				int max_id = 0;
				for (unsigned int hNr = 0; hNr < h.size(); hNr++) {
					if (max_id < h[hNr].id)
						max_id = h[hNr].id;
				}
				id2h.resize(max_id + 1);
				for (unsigned int hNr = 0; hNr < h.size(); hNr++) {
					id2h[h[hNr].id] = hNr;
				}
			}

			// preparing the variables
			int nlabels = crDetect.GetNumLabels();
			vector<float> max_heights(nlabels, 0.0f);
			vector<float> max_widths(nlabels, 0.0f);
			for ( int l = 0 ; l < nlabels ; ++l ) {
				max_heights[l] = max_height;
				max_widths[l] = max_width;
			}
			std::vector<float> kwidth;
			kwidth.resize(3, 0.0f);
			kwidth[0] = kernel_width[0];// kernel radius for smoothing voting space
			kwidth[1] = kernel_width[1];// kernel radius for gathering votes
			kwidth[2] = kernel_width[2];// sigma of the gaussian
			// after reading the images and check the already detected candidates

			//1. assign the right clusters to them
			vector<vector<Mat> > vImgAssign;
			crDetect.fullAssignCluster(img, depth_img, vImgAssign, scales);
			vector<vector<float> > candidates;

			vector<vector<Mat> > classConfidence;

			crDetect.getClassConfidence(vImgAssign, classConfidence);

			long int nvotes = 0;
			long int nvotes_class = 0;
			long int nvotes_oldclass = 0;

			std::vector<long int> nvotes_per_class(nlabels - 1, 0);

			for (unsigned int cNr = 0; cNr < nlabels - 1; cNr++) {
				std::vector<std::vector<float > > temp_candidates;
				long int tmpNvotes = 0;

				std::vector<float> scales_this_class(scales_per_class[cNr]);
				for (unsigned int i = 0; i < scales_this_class.size(); i++)
					scales_this_class[i] = scales[i];

				tmpNvotes = 0;
				float threshold_this_class = threshold_hierarchy / float(nlabels);
				if (do_hierarchy) {
					// you can multiply this threshold by the linkage weight of a class parent
					threshold_this_class = threshold_this_class * h[h[id2h[cNr]].parent].linkage;
				}

				crDetect.detectPyramidMR(vImgAssign, temp_candidates, scales_this_class, ratios, kwidth, max_candidates, cNr, theta, threshold_this_class, classConfidence, tmpNvotes); //vImgDetect,;
				nvotes_class += tmpNvotes;

				nvotes_per_class[cNr] = tmpNvotes;

				for (unsigned int candNr = 0; candNr < temp_candidates.size(); candNr++)
					candidates.push_back(temp_candidates[candNr]);
			}

			// sorting the candidates based on their weight
			bool end_sort = false;
			if (candidates.size() < 2)
				end_sort = true;// we do not need sorting


			while (!end_sort) {
				end_sort = true;
				for (unsigned int i = 0; i < candidates.size() - 1; i++ ) {
					if (candidates[i][0] < candidates[i + 1][0]) {
						end_sort = false;
						std::vector<float> cand_temp;
						cand_temp = candidates[i];
						candidates[i] = candidates[i + 1];
						candidates[i + 1] = cand_temp;
					}
				}

			}

			//initializing the file for the candidates
			std::vector< std::vector<float> > boundingboxes(candidates.size());
			for (unsigned int candNr = 0; candNr < candidates.size(); candNr++) {
				int scNr = 0;
				while (candidates[candNr][3] != scales[scNr])
					scNr++;

				bool form_candidates = true;
				if (form_candidates) {
					Candidate cand(crDetect.GetCRForest(), img, candidates[candNr], candNr, do_bpr);
					crDetect.voteForCandidate(vImgAssign[scNr], cand, kernel_width[0], ratios, max_widths[candidates[candNr][4]], max_heights[candidates[candNr][4]]);
					cand.getBBfromBpr(); // bounding box estimation
					boundingboxes[candNr] = cand.bb;
				}
			}

			// printing the candidate file
			sprintf_s(buffer, "%s/candidates.txt", buffer2);
			std::ofstream fp_cands;
			fp_cands.open(buffer);
			fp_cands << candidates.size() << std::endl;
			for (unsigned int candNr = 0; candNr < candidates.size(); candNr++) {
				fp_cands << candidates[candNr][0] << " " << candidates[candNr][1] << " " << candidates[candNr][2] << " " << candidates[candNr][3] << " " << candidates[candNr][4] << " " << candidates[candNr][5] << std::endl;
			}
			fp_cands << nvotes_class << std::endl;
			fp_cands << nvotes_per_class.size();
			for (unsigned int cNr = 0; cNr < nvotes_per_class.size(); cNr++) {
				fp_cands << " " << nvotes_per_class[cNr];
			}
			fp_cands << std::endl;
			fp_cands.close();

			// printing the bounding boxes file
			sprintf_s(buffer, "%s/boundingboxes.txt", buffer2);
			std::ofstream fp_boxes;
			fp_boxes.open(buffer);
			fp_boxes << boundingboxes.size() << std::endl;
			for (unsigned int boxNr = 0; boxNr < boundingboxes.size(); boxNr++) {
				fp_boxes << boundingboxes[boxNr][0] << " " << boundingboxes[boxNr][1] << " " << boundingboxes[boxNr][2] << " " << boundingboxes[boxNr][3] << std::endl;
			}
			fp_boxes.close();
		}
	}
}

// Init and start detector
void run_detect() {

	// Init forest with number of trees
	CRForest crForest( ntrees );

	// Load forest
	crForest.loadForest(treepath.c_str(), off_tree);

	std::vector<int> temp_classes; temp_classes.resize(1); temp_classes[0] = -1;
	crForest.SetTrainingLabelsForDetection(temp_classes);

	// Init detector
	CRForestDetector crDetect(&crForest, p_width, p_height, -1.0, -1.0, do_bpr);
	nlabels = crForest.GetNumLabels();

	// create directory for output
	string execstr = "mkdir ";
	execstr += outpath;

	system( execstr.c_str() );

	// run detector
	if (do_hierarchy) {
		crForest.loadHierarchy(hierarchy.c_str(), off_tree);
	}
	detect(crDetect);
}

/*! \brief Brief description.
 *         Brief description continued.
 *
 *  Detailed description starts here.
 */
int main(int argc, char *argv[]) {
	int mode = 1;

	// Check argument
	if (argc < 2) {
		cout << endl << endl << endl;
		cout << "Usage: CRForest-Detector[.exe] mode [config.txt] arguments" << std::endl;
		cout << endl << endl ;
		cout << "Training" << endl;
		cout << "  mode = 0; " << std::endl;
		cout << "  arguments: " << std::endl;
		cout << "    [number_of_trees]  [tree_offset=0]" << endl << endl;
		cout << "  These parameters are for parallelization and can be ignored for serial training of trees." << endl;
		cout << "  [number_of_trees]:  the number of trees to train at this run." << endl;
		cout << "  [tree_offset]:  The offset for saving the trained trees. For only training the 7th tree: number_of_trees = 1 and tree_offset = 6" << endl;
		cout << endl << endl;

		cout << "Detection " << endl;
		cout << "  mode = 1; " << std::endl;
		cout << "  arguments: " << std::endl;
		cout << "    [number_of_trees] [test_image_offset] [number_of_test_images] [tree_offset] [test_class] [test_set] [test_scale] " << std::endl;
		cout << endl;
		cout << "  [number_of_trees] replaces the ntrees in the config file " << endl;
		cout << "  [test_image_offset] begin detection at this image instead of the first " << endl;
		cout << "  [number_of_test_images] run detection on this number of images instead of doing it for all images" << endl;
		cout << "  [tree_offset] loading the trees starting from this offset" << endl;
		cout << "  [test_class] running the detection only on this class" << endl;
		cout << "  [test_set] running the detection on images only in one test set" << endl;
		cout << "  [test_scale] running the detection only at this scale instead of all scales" << endl;
		cout << endl << endl << endl ;
	} else {

		std::cout << "number of arguments" << argc << endl;

		if (argc > 1)
			mode = atoi(argv[1]);
		else
			mode = 2;

		off_tree = 0;

		// load configuration for dataset
		if (argc > 2)
			loadConfig(argv[2], mode);
		else
			loadConfig("config.txt", mode);

		float test_scale = -1;
		switch ( mode ) {
		case 0:
			// train forest
			if (argc > 3)
				ntrees = atoi(argv[3]);

			if (argc > 4)
				off_tree = atoi(argv[4]);
			scale_tree = 1.0f;

			cout << endl;
			cout << "training mode " << training_mode << " ntrees " << ntrees << " tree_offset " << off_tree  << endl;
			run_train();
			break;

		case 1: // detection
			cout << "running the detection" << endl;
			test_num = -1;
			off_test = 0;

			if (argc > 3)
				ntrees = atoi(argv[3]);

			if (argc > 4) {
				off_test = atoi(argv[4]);
				test_num = 1;
			}

			if (argc > 5)
				test_num = atoi(argv[5]);

			off_tree = 0;
			if (argc > 6)
				off_tree = atoi(argv[6]);

			select_test_class = 0;
			if (argc > 7)
				select_test_class = atoi(argv[7]);

			select_test_set = -1;
			if (argc > 8)
				select_test_set = atoi(argv[8]);

			if (argc > 9) { // test_scale
				test_scale = atof(argv[9]);
				if (test_scale > 0.0f) {
					scales.resize(1);
					scales[0] = test_scale;
				}
			}
			run_detect();
			break;

		default:
			std::cout << " The default mode is not defined" << std::endl;
			break;

		}
	}

	return 0;
}

