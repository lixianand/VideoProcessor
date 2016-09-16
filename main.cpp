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

void overlayImage(Mat* overlay, Mat* src, double transparency)
{
    for (int i = 0; i < src->cols; i++) {
        for (int j = 0; j < src->rows; j++) {
            Vec3b &intensity = src->at<Vec3b>(j, i);
            Vec3b &intensityoverlay = overlay->at<Vec3b>(j, i);
            for(int k = 0; k < src->channels(); k++) {
                uchar col = intensityoverlay.val[k];
                if (col != 0){
                    intensity.val[k] = (intensity.val[k]*(1-transparency) + (transparency*col));
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
	//Create cheap log file
	std::ofstream out("log.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());

    const string infilepath = "C:\\Users\\nag1\\Desktop\\Programming\\c_cpp\\Codeblocks_Projects\\VideoProcessor\\readfile.avi";
    VideoCapture capture(infilepath);
    const string outfilepath3 = "C:\\Users\\nag1\\Desktop\\Programming\\c_cpp\\Codeblocks_Projects\\VideoProcessor\\writefile3.avi";
    const string outfilepath4 = "C:\\Users\\nag1\\Desktop\\Programming\\c_cpp\\Codeblocks_Projects\\VideoProcessor\\writefile4.avi";
    cout << to_string(capture.get(CAP_PROP_FRAME_HEIGHT )) << "x" << to_string(capture.get(CAP_PROP_FRAME_WIDTH)) <<
        " at " << to_string(capture.get(CAP_PROP_FPS)) << " FPS" << endl;
    cout << "Total " << to_string(capture.get(CAP_PROP_FRAME_COUNT)) << " frames" << endl;
    VideoWriter cannyout(outfilepath3,CV_FOURCC('D','I','V','X'),capture.get(CAP_PROP_FPS),
                     Size(capture.get(CAP_PROP_FRAME_WIDTH ),capture.get(CAP_PROP_FRAME_HEIGHT)),true);
    VideoWriter output(outfilepath4,CV_FOURCC('D','I','V','X'),capture.get(CAP_PROP_FPS),
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

		Polygon polygon { cv::Point(0,0) };
		Mat cannyimage;
		ProcessImage( frame, cannyimage, polygon );
		AveragePolygon ( polygon,  polygons, 3, 5);
		std::vector<cv::Point> vecpolygon;
		vecpolygon.push_back(polygon[3]);
		vecpolygon.push_back(polygon[2]);
		vecpolygon.push_back(polygon[1]);
		vecpolygon.push_back(polygon[0]);
		int timeposition{static_cast<int>((i/capture.get(CAP_PROP_FPS)))};
		cout << to_string(timeposition) << "s ";
		PrintLanes(polygon);
		
		//Overlay lanes
		cv::Mat polygonimage{ frame.size(), frame.type(), cv::Scalar(0) };
		cv::Point newpolygon[4];
		std::copy( polygon.begin(), polygon.end(), newpolygon );
		cv::fillConvexPoly( polygonimage, newpolygon, 4, Scalar(0,255,0,127) );
		overlayImage( &polygonimage, &frame, 0.5);
		overlayImage( &polygonimage, &cannyimage, 0.5);

        double percent = 100*i/capture.get(CAP_PROP_FRAME_COUNT);
		if ( i%100 == 0 ) cout << to_string(percent) << "% done" << endl;
		
		cannyout << cannyimage;
        output << frame;
        //imshow("w", frame);
        //waitKey(0); // waits to display frame
        //if ( frames >= 300) break;

    }
    high_resolution_clock::time_point t2(high_resolution_clock::now());
    double spd = duration_cast<milliseconds>(t2 - t1).count() / frames;
    // releases and window destroy are automatic in C++ interface
    std::cout.rdbuf(coutbuf); //reset to standard output again
    cout << "Completed, it took " << to_string(spd) << " ms per frame.";
    cannyout.release();
    output.release();
}
