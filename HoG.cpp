/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <vector>
#include <iostream>
#include "HoG.h"



HoG::HoG() {
	bins = 9;
	binsize = (3.14159265f * 80.0f) / float(bins);;

	g_w = 5;
	Gauss = cvCreateMat( g_w, g_w, CV_32FC1 );
	double a = -(g_w - 1) / 2.0;
	double sigma2 = 2 * (0.5 * g_w) * (0.5 * g_w);
	double count = 0;
	for (int x = 0; x < g_w; ++x) {
		for (int y = 0; y < g_w; ++y) {
			double tmp = exp(-( (a + x) * (a + x) + (a + y) * (a + y) ) / sigma2);
			count += tmp;
			cvSet2D( Gauss, x, y, cvScalar(tmp) );
		}
	}
	cvConvertScale( Gauss, Gauss, 1.0 / count);

	ptGauss = new float[g_w * g_w];
	int i = 0;
	for (int y = 0; y < g_w; ++y)
		for (int x = 0; x < g_w; ++x)
			ptGauss[i++] = (float)cvmGet( Gauss, x, y );

}


void HoG::extractOBin(Mat &Iorient, Mat &Imagn, vector<Mat> &out) {
	double *desc = new double[bins];

	// reset output image (border=0) and get pointers
	uchar **ptOut     = new uchar*[bins];
	uchar **ptOut_row = new uchar*[bins];
	for (int k = 0; k < bins; ++k) {
		ptOut[k] = out[k].data;
	}

	// get pointers to orientation, magnitude
	int step = Iorient.step[0];
	uchar *ptOrient = Iorient.data;
	uchar *ptOrient_row;

	uchar *ptMagn = Imagn.data;
	uchar *ptMagn_row;

	int off_w = int(g_w / 2.0);
	for (int l = 0; l < bins; ++l)
		ptOut[l] += off_w * step;

	for (int y = 0; y < Iorient.rows - g_w; y++, ptMagn += step, ptOrient += step) {

		// Get row pointers
		ptOrient_row = &ptOrient[0];
		ptMagn_row = &ptMagn[0];
		for (int l = 0; l < bins; ++l)
			ptOut_row[l] = &ptOut[l][0] + off_w;

		for (int x = 0; x < Iorient.cols - g_w; ++x, ++ptOrient_row, ++ptMagn_row) {

			calcHoGBin( ptOrient_row, ptMagn_row, step, desc );

			for (int l = 0; l < bins; ++l) {
				*ptOut_row[l] = (uchar)desc[l];
				++ptOut_row[l];
			}
		}

		// update pointer
		for (int l = 0; l < bins; ++l)
			ptOut[l] += step;
	}

	delete[] desc;
	delete[] ptOut;
	delete[] ptOut_row;
}



