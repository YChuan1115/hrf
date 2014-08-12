cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(hf_mc)
set(CMAKE_BUILD_TYPE Release)


find_package(PCL 1.7 REQUIRED)
find_package(Boost 1.56.0 REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(Eigen3 3.2 REQUIRED)


## custom OpenCV
set(OpenCV_INCLUDE_DIRS "/home/stfn/libs/opencv-2.4.9/include" "/home/stfn/libs/opencv-2.4.9/include/opencv" "/home/stfn/libs/opencv-2.4.9/include/opencv2")
set(OpenCV_LIBRARY_DIRS "/home/stfn/libs/opencv-2.4.9/build/lib")
set(OpenCV_LIBS "opencv_core;opencv_highgui;opencv_imgproc;opencv_gpu;opencv_objdetect")

# custom Eigen3
set(EIGEN_INCLUDE_DIRS "/home/stfn/libs/eigen3.2.1")

# custom Boost
set (BOOST_INCLUDE_DIRS "/home/stfn/libs/boost_1_56_0")
set (BOOST_LIBRARY_DIRS "/home/stfn/libs/boost_1_56_0/stage/lib")
set (BOOST_LIBS "boost_timer")

include_directories(
	${BOOST_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${BOOST_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (CRForest-Detector CRForest-Detector.cpp Candidate.cpp CRForestDetector.cpp CRPatch.cpp CRTree.cpp HoG.cpp)
add_executable (untitled untitled.cpp)
add_executable (hog hog.cpp)
add_executable (myhog myhog.cpp HoG.cpp GpuHoG.cpp)


target_link_libraries (CRForest-Detector 
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
	${BOOST_LIBS}
)

target_link_libraries (untitled 
	${OpenCV_LIBS}
)

target_link_libraries (hog 
	${OpenCV_LIBS}
)

target_link_libraries (myhog 
	${OpenCV_LIBS}
)


