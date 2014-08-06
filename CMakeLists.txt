cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(hf_mc)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED )
#find_package(Eigen3 3.2 REQUIRED)

include_directories(
	"~/eigen3.2.1"
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
#	${EIGEN_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (CRForest-Detector CRForest-Detector.cpp Candidate.cpp CRForestDetector.cpp CRPatch.cpp CRTree.cpp HoG.cpp)

target_link_libraries (CRForest-Detector 
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
)


set(CMAKE_BUILD_TYPE Release)