#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cxcore.h>
#include <cv.h>


// save floating point images
inline void saveFloatImage( char* buffer , IplImage* img){
	std::ofstream fp_out;
	fp_out.open(buffer);

	float x;
	x = float(img->width);
	fp_out.write(reinterpret_cast<char *>(&x),sizeof(float));
	x = float(img->height);
	fp_out.write(reinterpret_cast<char *>(&x),sizeof(float));
	
	int stepData;
	float* rawData;
	cvGetRawData( img, (uchar**)&(rawData), &stepData);
	stepData /= sizeof(rawData[0]);

	for (int cy = 0; cy < img->height; cy++){		
		for (int cx = 0; cx < img->width; cx++){
			x = *(rawData + cx + cy*stepData);
			fp_out.write(reinterpret_cast<char *>(&x),sizeof(float));
		}
	} 

	fp_out.close();
}

inline bool isInsideRect(CvRect* rect, int x, int y){
	
	if (x > rect->x && x < rect->x + rect->width && y > rect->y && y < rect->y + rect->height){
		return true;
	}else{
		return false;
	}
}

inline bool isInsideKernel2D(float x, float y, float cx, float cy , float radius){

	float sum = (x-cx)*(x-cx) + (y-cy)*(y-cy);

	if (sum > radius*radius) 
		return false;
	
	return true;
}


template <class myType>
inline void print_vector(std::vector<myType>& vec){
	for (unsigned int sz= 0; sz < vec.size(); sz++)
		std::cout << " " << vec[sz];
	std::cout<< " " << std::endl;
}
