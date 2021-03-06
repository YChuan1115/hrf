#include "CRForestDetector.hpp"
#include "myutils.hpp"

#define timer fubar
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>

#include <highgui.h>
#include "opencv2/gpu/gpu.hpp"

#include <tbb/task_group.h>

#include <vector>


using namespace std;
using namespace cv;



int COUNT;

// given the cluster assignment images, we are voting into the voting space vImgDetect
void CRForestDetector::voteColor(Mat &depth_img, vector<Mat> &vImgAssign, vector<Mat> &vImgDetect, vector<Mat> &classProbs, float xShift, float yShift, int this_class, Rect &focus, float prob_threshold) {
	// vImgDetect are all initialized before

	if (vImgAssign.size() < 1)
		return;

	// compensate for the shift based on the
	if (xShift < 0)
		xShift = vImgAssign[0].cols * 0.50;
	if (yShift < 0)
		yShift = vImgAssign[0].rows * 0.50;

	float ntrees = float(vImgAssign.size());
	tbb::task_group tbb_tg;

	// loop over trees
	for (size_t trNr = 0; trNr < vImgAssign.size(); ++trNr) {

		function<void ()> process = [ &, trNr]() {
			// loop over assign height
			for (int cy = 0 ; cy < vImgAssign[trNr].rows; ++cy) {
				float *ptr = vImgAssign[trNr].ptr<float>(cy);

				// loop over assign width
				for (int cx = 0; cx < vImgAssign[trNr].cols; ++cx) {
					// get the leaf_id
					if (ptr[cx] < 0)
						continue;
					float depth_scale = depth_img.ptr<float>(cy)[cx];
					if (depth_scale < 0.1f)
						depth_scale = 1.f;

					LeafNode *tmp = crForest_->vTrees_[trNr]->getLeaf(ptr[cx]);

					// loop over labels
					for (size_t lNr = 0; lNr < vImgDetect.size(); ++lNr) {

						if ((this_class >= 0 ) && (this_class != lNr)) // the voting should be done on a single class only
							continue;

						bool condition;
						if (prob_threshold < 0) {
							condition = (class_ids_[trNr][lNr] > 0 && tmp->vPrLabel[lNr] * class_ids_[trNr].size() > 1);
						} else {
							condition = (class_ids_[trNr][lNr] > 0  &&  classProbs[lNr].ptr<float>(cy)[cx] > prob_threshold);
						}

						if (condition) {
							// vote for all points stored in a leaf
							float w = tmp->vPrLabel[lNr] / ntrees;
							vector<float>::const_iterator itW = tmp->vCenterWeights[lNr].begin();
							for (vector<Point>::const_iterator it = tmp->vCenter[lNr].begin(); it != tmp->vCenter[lNr].end(); ++it, itW++) {
								// calc object hypothesis center
								int x = int(float(cx) - float((*it).x) / depth_scale + 0.5 + xShift);
								int y = int(float(cy) - float((*it).y) / depth_scale + 0.5 + yShift);


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
		};

		tbb_tg.run(bind(process));
	}
	tbb_tg.wait();
}


// Gathering the information in the support of each candidate
void CRForestDetector::voteForCandidate(Mat &depth_img, vector<Mat> &vImgAssign, Candidate &cand, int kernel_width, float max_width, float max_height ) {

	if (vImgAssign.size() < 1)
		return;

	time_t t = time(NULL);
	int seed = (int)t;//1407685013;
	CvRNG pRNG(seed);
	double value = 0.0;
	double sample_votes = 0.998;// ignore the patches by this probability

	// get cand_x and cand_y at this scale
	int cand_x = int(cand.x_ * cand.scale_ + 0.5);
	int cand_y = int(cand.y_ * cand.scale_ + 0.5);

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
	for (size_t trNr = 0; trNr < int(ntrees); trNr++) {

		// looping over all locations within candidate roi
		for (int cy = min_y; cy < max_y; ++cy) {
			float *ptr = vImgAssign[trNr].ptr<float>(cy);

			for (int cx = min_x; cx < max_x; ++cx) {

				value = cvRandReal(&pRNG);

				if (value < sample_votes || ptr[cx] < 0)
					continue;

				LeafNode *tmp = crForest_->vTrees_[trNr]->getLeaf(ptr[cx]);
				float w = tmp->vPrLabel[cand.c_] / ntrees;
				if (w < 0.0e-7)
					continue;

				float depth_scale = depth_img.ptr<float>(cy)[cx];
				if (depth_scale < 0.1f)
					depth_scale = 1.f;

				float w_element = 0.0f;
				int idNr = 0;
				vector<float>::const_iterator itW = tmp->vCenterWeights[cand.c_].begin();
				for (vector<Point>::const_iterator it = tmp->vCenter[cand.c_].begin() ; it != tmp->vCenter[cand.c_].end(); ++it, ++idNr, itW++) {
					int x = int(float(cx) - float((*it).x) / depth_scale + 0.5);
					int y = int(float(cy) - float((*it).y) / depth_scale + 0.5);

					float squared_dist = (x - cand_x) * (x - cand_x) + (y - cand_y) * (y - cand_y);
					if (squared_dist < kernel_width * kernel_width) {
						w_element += w * (*itW);
					}
				}

				if ( w_element > 0.0 && cand.do_bpr_) {
					// update the backprojection image
					cand.backproj_mask_.ptr<float>(cy)[cx] += w_element;
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
	vector<vector<gpu::GpuMat> > smoothAcc;
	smoothAcc.resize(scales.size());
	for (size_t scNr = 0; scNr < scales.size(); ++scNr) {
		int adapKwidth = int(kernel_width * scales[scNr] / 2.0f) * 2 + 1;
		float adapKstd  = kernel_std * scales[scNr];

		smoothAcc[scNr].resize(imgDetect[scNr].size());
		for (int cNr = 0; cNr < imgDetect[scNr].size(); ++cNr) {
			if ((this_class >= 0) && ( this_class != cNr))
				continue;
			gpu::GpuMat gpu_imgDetect(imgDetect[scNr][cNr]);
			if (adapKwidth > 31) {
				adapKwidth = 31;
			}
			gpu::GaussianBlur(gpu_imgDetect, smoothAcc[scNr][cNr], Size(adapKwidth, adapKwidth), adapKstd);
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
	for (int count = 0; candNr < max_cands ; ++count) { // count can go until infinity
		bool flag = false;
		vector<float> max_position(6, -1); // max_val, x, y, scNr, cNr, rNr
		// detect the maximum
		for (size_t scNr = 0; scNr < scales.size(); ++scNr) {
			if (shift < 0.0f) {
				xShift = imgDetect[scNr][default_class].cols * 0.25;
				yShift = imgDetect[scNr][default_class].rows * 0.25;
			} else {
				xShift = imgDetect[scNr][default_class].cols * shift;
				yShift = imgDetect[scNr][default_class].rows * shift;
			}
			for (size_t cNr = 0; cNr < imgDetect[scNr].size(); ++cNr) {
				if ((this_class >= 0) && ( this_class != cNr))
					continue;

				gpu::minMaxLoc(smoothAcc[scNr][cNr], &min_val_temp, &max_val_temp, &min_loc_temp, &max_loc_temp);
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
		for (size_t scNr = 0; scNr < scales.size(); scNr++) {
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

			for (int cNr = 0; cNr < imgDetect[scNr].size(); ++cNr) {
				if (max_position[4] >= 0 && cNr != max_position[4])
					continue;

				// clear candidates bounding box
				smoothAcc[scNr][cNr](Rect(x, y, rwidth, rheight)) = Scalar(0.0);
			}
		}
	}
}

void CRForestDetector::detectPyramidMR(vector<vector<Mat> > &vImgAssign, vector<vector<float> > &candidates, vector<float> &scales, vector<float> &kernel_width, vector<float> &params, vector<vector<Mat> > &classProbs, Mat &depth_img) {
	int max_cands = params[0];
	int this_class = params[1];
	float threshold = params[2];
	float prob_threshold = params[3];
	tbb::task_group tbb_tg;

	Mat tmp_depth = depth_img.clone();
	Mat points(tmp_depth.size(), CV_32FC3);
	Mat dist(tmp_depth.size(), CV_32FC1);
	tmp_depth.convertTo(tmp_depth, CV_32FC1);
	calcPoints(tmp_depth, points, 1.0);
	for (int y = 0; y < points.rows; ++y) {
		Eigen::Vector3f *r_ptr_points = points.ptr<Eigen::Vector3f>(y);
		float *r_ptr_dist = dist.ptr<float>(y);
		for (int x = 0; x < points.cols; ++x) {
			r_ptr_dist[x] = r_ptr_points[x].norm();
		}
	}

	vector<vector<Mat> > vvImgDetect(vImgAssign.size());
	{
		boost::timer::auto_cpu_timer at;

		// accumulating votes for all scales and classes
		for (size_t scNr = 0; scNr < scales.size(); scNr++) {
			vvImgDetect[scNr].resize(crForest_->GetNumLabels());

			for (size_t lNr = 0; lNr < crForest_->GetNumLabels(); ++lNr) {
				if ( (this_class >= 0 ) && (this_class != lNr) )
					continue;

				vvImgDetect[scNr][lNr] = Mat::zeros(Size(vImgAssign[scNr][0].cols * 2.0f + 0.5, vImgAssign[scNr][0].rows * 2.0f + 0.5), CV_32FC1);
			}

			//voteColor(vImgAssign[scNr], vvImgDetect[scNr], classProbs[scNr], -1, -1, this_class, default_rect__, prob_threshold);
			function<void(void)> job_func = bind(&CRForestDetector::voteColor, this, ref(dist), ref(vImgAssign[scNr]), ref(vvImgDetect[scNr]), ref(classProbs[scNr]), -1, -1, this_class, default_rect__, prob_threshold);
			tbb_tg.run(job_func);
		}
		tbb_tg.wait();
		cout << "\t voteColor: ";
	}

#if 0
	// save the Hough spaces
	for (unsigned int scNr = 0; scNr < scales.size(); scNr++) {
		cv::Mat hough = vvImgDetect[scNr][this_class].clone();
		cv::normalize(hough, hough, 0, 255, cv::NORM_MINMAX);
		hough.convertTo(hough, CV_8UC1);
		std::stringstream ss;
		ss << "hough_c" << this_class << "_s" << scNr << ".png";
		cv::imwrite(ss.str(), hough);
	}

#endif

	boost::timer::auto_cpu_timer at;
	// detecting the peaks in the voting space
	detectPeaks(candidates, vvImgDetect, scales, max_cands, kernel_width[0] , kernel_width[2], true, -1, this_class, threshold);
	cout << "\t detectPeaks: ";
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


	// x,y top left; cx,cy center of patch
	int xoffset = patch_size_.width / 2;
	int yoffset = patch_size_.height / 2;

	tbb::task_group tbb_tg;

	for (int y = 0; y < img.rows - patch_size_.height; ++y) {

		function<void ()> process = [ &, y]() {

			for (int x = 0; x < img.cols - patch_size_.width; ++x) {
				vector<const LeafNode *> result;

				crForest_->regression(result, vImg, x, y);

				for (size_t treeNr = 0; treeNr < result.size(); treeNr++) {
					vImgAssign[treeNr].ptr<float>(y + yoffset)[x + xoffset] = float(result[treeNr]->idL);
				}

			} // end for x
		};


		tbb_tg.run(bind(process));
		if (y % 50 == 0)
			tbb_tg.wait();


	} // end for y
	tbb_tg.wait();
}


void CRForestDetector::fullAssignCluster(Mat &img, Mat &depth_img, vector<vector<Mat> > &vvImgAssign, vector<float> &scales) {

	tbb::task_group tbb_tg;
	size_t ntrees = crForest_->vTrees_.size();
	Scalar vvImgAssignValue(-1.0);
	vector<Mat> img_scaled(scales.size());
	vector<Mat> depth_scaled(scales.size());

	vvImgAssign.resize(scales.size());

	for (size_t scaleNr = 0; scaleNr < scales.size(); scaleNr++) {
		vvImgAssign[scaleNr].resize(ntrees);
		// rescaling the image to scales[scaleNr]
		resize( img, img_scaled[scaleNr], Size(img.cols * scales[scaleNr] + 0.5, img.rows * scales[scaleNr] + 0.5), 0, 0, CV_INTER_LINEAR );
		resize( depth_img, depth_scaled[scaleNr], Size(depth_img.cols * scales[scaleNr] + 0.5, depth_img.rows * scales[scaleNr] + 0.5), 0, 0, CV_INTER_NN );

		//looping over the trees in the forest
		for (int treeNr = 0; treeNr < ntrees; treeNr++) {
			vvImgAssign[scaleNr][treeNr] = Mat(Size(img.cols * scales[scaleNr] + 0.5, img.rows * scales[scaleNr] + 0.5), CV_32FC1);
			vvImgAssign[scaleNr][treeNr] = vvImgAssignValue;
		}

		//assignCluster(img_scaled[scaleNr], depth_scaled[scaleNr], vvImgAssign[scaleNr], scales[scaleNr]);
		function<void(void)> job_func = bind(&CRForestDetector::assignCluster, this, ref(img_scaled[scaleNr]), ref(depth_scaled[scaleNr]), ref(vvImgAssign[scaleNr]), ref(scales[scaleNr]));
		tbb_tg.run(job_func);
	}

	tbb_tg.wait();
}


// Getting the per class confidences TODO: this has to become scalable
void CRForestDetector::getClassConfidence(vector<vector<Mat> > &vImgAssign, vector<vector<Mat> > &classConfidence) {
	tbb::task_group tbb_tg;
	int nlabels = crForest_->GetNumLabels();

	// allocating space for the classConfidence
	classConfidence.resize(vImgAssign.size());
	for (size_t i = 0; i < vImgAssign.size(); i++) {
		classConfidence[i].resize(nlabels);
		for (int j = 0; j < nlabels; j++) {
			classConfidence[i][j] = Mat::zeros(Size(vImgAssign[i][0].cols, vImgAssign[i][0].rows), CV_32FC1);
		}
	}

	// looping over the scales
	for (size_t scNr = 0; scNr < vImgAssign.size(); ++scNr) {
		function<void(void)> job_func = bind(&CRForestDetector::getClassConfidencePerScale, this, ref(vImgAssign[scNr]), ref(classConfidence[scNr]), nlabels);
		tbb_tg.run(job_func);
	}

	tbb_tg.wait();

#if 0
	// save the class confidence
	for (size_t i = 0; i < vImgAssign.size(); i++) {
		for (int j = 0; j < nlabels; j++) {
			cv::Mat cc = classConfidence[i][j].clone();
			cv::normalize(cc, cc, 0, 255, cv::NORM_MINMAX);
			cc.convertTo(cc, CV_8UC1);
			std::stringstream ss;
			ss << "cc_c" << j << "_s" << i << ".png";
			cv::imwrite(ss.str(), cc);
		}
	}

#endif
}

void CRForestDetector::getClassConfidencePerScale(vector<Mat> &vImgAssign, vector<Mat> &classConfidence, int nlabels) {
	// TODO: why the tmpClassProbs?

	int w = vImgAssign[0].cols;
	int h = vImgAssign[0].rows;

	// function variables
	int outer_window = 8; // TODO: this parameter shall move to the inputs.
	float inv_tree = 1.0f / vImgAssign.size();

	// looping over the trees
	for (size_t trNr = 0; trNr < vImgAssign.size() ; trNr++) {
		// here make a temporary structure of all the probabilities and then smooth it with a kernel.
		vector<Mat> tmpClassProbs(nlabels);
		for (int cNr = 0; cNr < nlabels; ++cNr) {
			tmpClassProbs[cNr] = Mat::zeros(Size(vImgAssign[trNr].cols, vImgAssign[trNr].rows), CV_32FC1);
		}

		for (int y = 0; y < h ; ++y) {
			for (int x = 0; x < w; ++x) {
				int leaf_id = vImgAssign[trNr].ptr<float>(y)[x];
				if (leaf_id < 0)
					continue;

				LeafNode *tmp = crForest_->vTrees_[trNr]->getLeaf(leaf_id);

				for (int cNr = 0; cNr < nlabels; ++cNr) {
					tmpClassProbs[cNr].ptr<float>(y)[x] = tmp->vPrLabel[cNr] * inv_tree;
				}
			}
		}

		// now values of the tmpClassProbs are set we can blur it to get the average
		for (int cNr = 0; cNr < nlabels; cNr++) {
			blur(tmpClassProbs[cNr], tmpClassProbs[cNr], Size(outer_window, outer_window));
		}

		for (int cNr = 0; cNr < nlabels; cNr++) {
			//  LOOPING OVER ALL PIXELS
			for (int y = 0; y < h; y++) {
				for (int x = 0 ; x < w; x++) {
					classConfidence[cNr].ptr<float>(y)[x] += tmpClassProbs[cNr].ptr<float>(y)[x];
				}
			}
		}
	}
}