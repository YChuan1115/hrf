cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(hf_mc)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")


#find_package(Boost 1.56.0 COMPONENTS system thread timer)
#find_package(OpenCV REQUIRED)
#find_package(Eigen3 3.2 REQUIRED)
find_package(PCL 1.7 REQUIRED)


# custom Boost
if(NOT BOOST_FOUND OR PCL_FOUND)
	set (BOOST_ROOT "/home/stfn/libs/boost_1_56_0")
	set (BOOST_INCLUDE_DIRS "/home/stfn/libs/boost_1_56_0")
	set (BOOST_LIBRARY_DIRS "/home/stfn/libs/boost_1_56_0/stage/lib")
	set (BOOST_LIBS "boost_timer;boost_thread;boost_system")
endif()

## custom OpenCV
set(OpenCV_INCLUDE_DIRS "/home/stfn/libs/opencv-2.4.9/include" "/home/stfn/libs/opencv-2.4.9/include/opencv" "/home/stfn/libs/opencv-2.4.9/include/opencv2" "/home/stfn/libs/opencv-2.4.9/modules/*/include")
set(OpenCV_LIBRARY_DIRS "/home/stfn/libs/opencv-2.4.9/build/lib")
set(OpenCV_LIBS "opencv_core;opencv_highgui;opencv_imgproc;opencv_gpu;opencv_objdetect")

# custom Eigen3
set(EIGEN_INCLUDE_DIRS "/home/stfn/libs/eigen3.2.1")



include_directories(
	${EIGEN_INCLUDE_DIRS}
	${BOOST_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

link_directories(${BOOST_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})


add_executable (CRForest-Detector Candidate.cpp CRForest.cpp CRForest-Detector.cpp CRForestDetector.cpp CRForest.cpp CRPatch.cpp CRTree.cpp HoG.cpp)

target_link_libraries (CRForest-Detector 
	${OpenCV_LIBS}
	${BOOST_LIBS}
	tbb
)