#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <math.h>
#include <cmath>
#include <cfloat>
#include "lane_detect_processor.h"

using namespace cv;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

string RenameOutputFile (const string& str)
{
  size_t foundslash = str.find_last_of("/\\");
  size_t founddot = str.find_last_of(".");  
  return str.substr(foundslash + 1, (founddot - foundslash - 1)) + "_edit.avi";
}

void PrintLanes(Polygon& polygon) {
	if (polygon[0] == cv::Point(0,0)) { cout << endl; return; }
	int characters{80};
	int width{800};
	int lposition{characters*(polygon[0].x)/width};
	int rposition{characters*(polygon[1].x)/width};
	int middle{(lposition+rposition)/2};
	cout << "|";
	for (int i = 0; i < characters; i++) {
		if (i == lposition) {
			cout << "<";
		} else if (i == rposition){
			cout << ">";
		} else if (i == middle){
			cout << "x";
		} else {
			cout << "-";
		}
	}
	cout << "|" << endl;
}

void OverlayImage( cv::Mat* overlay,
                   cv::Mat* src )
{
    for (int i = 0; i < src->cols; i++) {
        for (int j = 0; j < src->rows; j++) {
			if ( overlay->at<uchar>(j, i) != 0 ) {
				cv::Vec3b &intensity = src->at<cv::Vec3b>(j, i);
				intensity.val[1] = (intensity.val[1] + 255) * 0.5f;
			}
        }
    }
}

int main(int argc,char *argv[])
{
	//Check arguments passed
	if (argc < 2) {
		std::cout << "No arguments passed, press ENTER to exit..." << std::endl;
		std::cin.get();
		return 0;
	}
	
	cv::namedWindow("Output", CV_WINDOW_NORMAL );
	//Create cheap log file
	std::ofstream out("log.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());

	for (int i = 1; i < argc; i++ ) {
		VideoCapture capture(argv[i]);
		const string outfilepath(RenameOutputFile(argv[1]));
		
		cout << to_string(capture.get(CAP_PROP_FRAME_HEIGHT )) << "x" << to_string(capture.get(CAP_PROP_FRAME_WIDTH)) <<
			" at " << to_string(capture.get(CAP_PROP_FPS)) << " FPS" << endl;
		cout << "Total " << to_string(capture.get(CAP_PROP_FRAME_COUNT)) << " frames" << endl;
		VideoWriter output(outfilepath,CV_FOURCC('D','I','V','X'),capture.get(CAP_PROP_FPS),
						 Size(capture.get(CAP_PROP_FRAME_WIDTH ),capture.get(CAP_PROP_FRAME_HEIGHT)),true);
		Mat frame;
		capture >> frame;

		if( !capture.isOpened() )
			throw "Error when reading steam_avi";

		high_resolution_clock::time_point t1(high_resolution_clock::now());

		std::deque<Polygon> polygons;
		
		int frames(0);
		for( int i =0; i < capture.get(CAP_PROP_FRAME_COUNT) - 1; i++  ){
			//Get frame
			frames++;
			capture >> frame;
			cv::Mat workingimage { frame };
			
			//std::cout << "Frame #" << i << std::endl;
			Polygon polygon { cv::Point(0,0) };
			ProcessImage( workingimage, polygon );
			AveragePolygon ( polygon,  polygons, 4, 7);
			std::vector<cv::Point> vecpolygon;
			vecpolygon.push_back(polygon[3]);
			vecpolygon.push_back(polygon[2]);
			vecpolygon.push_back(polygon[1]);
			vecpolygon.push_back(polygon[0]);
			int timeposition{static_cast<int>((i/capture.get(CAP_PROP_FPS)))};
			//cout << to_string(timeposition) << "s ";
			PrintLanes(polygon);

			//Overlay lanes
			if ( polygon[0] != cv::Point(0,0) ) {
				//std::cout << "Lanes found!" << std::endl;
				cv::Point cvpointarray[4];
				std::copy( polygon.begin(), polygon.end(), cvpointarray );
				cv::Mat polygonimage{ frame.size(),
									  CV_8UC1,
									  cv::Scalar(0) };
				cv::fillConvexPoly( polygonimage, cvpointarray, 4,  cv::Scalar(1) );
				OverlayImage( &polygonimage, &frame );
			}
			
			double percent = 100*i/capture.get(CAP_PROP_FRAME_COUNT);
			//if ( i%100 == 0 ) cout << to_string(percent) << "% done" << endl;
			output << frame;
			//std::cout << "----------------------------------------------------------" << std::endl;
			imshow("Output", frame);
			waitKey(1); // waits to display frame
			//if ( frames >= 300) break;

		}
		high_resolution_clock::time_point t2(high_resolution_clock::now());
		double spd = duration_cast<milliseconds>(t2 - t1).count() / frames;
		cout << "Completed, it took " << to_string(spd) << " ms per frame.";
		output.release();
	}
		std::cout.rdbuf(coutbuf); //reset to standard output again
}
