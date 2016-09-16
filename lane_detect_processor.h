#ifndef LANE_DETECT_PROCESSOR_HPP_INCLUDED
#define LANE_DETECT_PROCESSOR_HPP_INCLUDED

#include <deque>
#include <array>
#include "opencv2/opencv.hpp"

typedef std::array<cv::Point, 4> Polygon;

struct EvaluatedContour {
    const std::vector<cv::Point>* contour;
    cv::RotatedRect ellipse;
    float lengthwidthratio;
	float angle;
	//cv::Moments moment;
	//cv::Point center;
    cv::Vec4f fitline;
};

struct PolygonDifferences {
	Polygon polygon;
	float differencefromaverage;
};

void CreateKeypoints( const std::vector<std::vector<cv::Point>>& contours,
	std::vector<cv::KeyPoint>& keypoints );
void EvaluateSegment( const std::vector<cv::Point>& contour, const int imageheight,
	std::vector<EvaluatedContour>&	evaluatedsegments );
void ConstructFromBlobs( const std::vector<cv::KeyPoint>& keypoints,
	std::vector<std::vector<cv::Point>>& constructedcontours );
void ConstructFromSegmentAndBlob( const std::vector<EvaluatedContour>& evaluatedsegments,
	const std::vector<cv::KeyPoint>& keypoints, std::vector<std::vector<cv::Point>>&
	constructedcontours );
void ConstructFromSegments( const std::vector<EvaluatedContour>& evaluatedsegments,
	std::vector<std::vector<cv::Point>>& constructedcontours );
void SortContours( const std::vector<EvaluatedContour>& evaluatedsegments,
	const int imagewidth, std::vector<EvaluatedContour>& leftcontours,
	std::vector<EvaluatedContour>& rightcontours );
void FindPolygon( Polygon& polygon, const std::vector<cv::Point>& leftcontour,
	const std::vector<cv::Point>& rightcontour );
double ScoreContourPair( const Polygon& polygon, const int imagewidth,
	const EvaluatedContour& leftcontour, const EvaluatedContour& rightcontour );
void AveragePolygon ( Polygon& polygon, std::deque<Polygon>& pastpolygons,
	int samplestoaverage, int samplestokeep );
bool ComparePolygons(const PolygonDifferences& a, const PolygonDifferences& b);
//void ProcessImage ( cv::Mat image, Polygon& polygon );
void ProcessImage ( cv::Mat image, cv::Mat& cannyimage, Polygon& polygon );

#endif // LANE_DETECT_PROCESSOR_HPP_INCLUDED
