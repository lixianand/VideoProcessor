//standard libraries
#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <deque>
#include <algorithm>
#include <math.h>

//3rd party libraries
#include "opencv2/core/core.hpp"

//project libraries
#include "lane_detect_constants.h"
#include "lane_detect_processor.h"

#ifndef M_PI
    #define M_PI_4 0.78539816339
#endif
#ifndef M_PI
	#define M_1_PI 0.31830988618
#endif
#define DEGREESPERRADIAN 57.2957795131
#define POLYGONSCALING 0.1

namespace lanedetectconstants {
		
	uint16_t ksegmentellipseheight{6};
	float ksegmentanglewindow{75.0f};
	float ksegmentlengthwidthratio{1.6f};
	float ksegmentsanglewindow{45.0f};
	uint16_t kellipseheight{25};
	float kanglewindow{70.0f};
	float klengthwidthratio{7.00f};
    float kcommonanglewindow{40.0f};
	float klowestscorelimit{-FLT_MAX};
	//Only effective when scoring contour pairs
    uint16_t kminroadwidth {250};
    uint16_t kmaxroadwidth {450};
	uint16_t koptimumwidth {350};
	float kellipseratioweight{1.3f};
	float kangleweight{-2.2f};
	float kcenteredweight{-1.0f};
	float kwidthweight{-3.0f};
	float klowestpointweight{-2.0f};
	//Only effective when scoring by optimal polygon
	Polygon optimalpolygon{ cv::Point(100,400), cv::Point(540,400), cv::Point(340,250), cv::Point(300,250) };	
}

//Main function
void ProcessImage ( cv::Mat& image,
                    Polygon& polygon )
{
//-----------------------------------------------------------------------------------------
//Image manipulation
//-----------------------------------------------------------------------------------------
	//Change to grayscale
	cv::cvtColor( image, image, CV_BGR2GRAY );
	
	//Blur to reduce noise
    cv::blur( image, image, cv::Size(3,3) );
	
//-----------------------------------------------------------------------------------------
//Find contours
//-----------------------------------------------------------------------------------------
	//Auto threshold values for canny edge detection
    //double otsuthreshval = cv::threshold( image, image, 0, 255,
	//	CV_THRESH_BINARY | CV_THRESH_OTSU );
	//Canny edge detection
    cv::Canny(image, image, 40, 120, 3 );
    //cv::Canny(image, image, otsuthreshval * 0.5, otsuthreshval );
	//cv::cvtColor( image, cannyimage, CV_GRAY2BGR );
	std::vector<Contour> detectedcontours;
    std::vector<cv::Vec4i> detectedhierarchy;
    cv::findContours( image, detectedcontours, detectedhierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	//std::cout << "Contours found: " << detectedcontours.size() << std::endl;
	//Contours removed by position in function

	//ToDo - There's way more I could be doing:
		//Dilate? Grow?
		
//-----------------------------------------------------------------------------------------
//Evaluate contours
//-----------------------------------------------------------------------------------------	
	std::vector<EvaluatedContour> evaluatedchildsegments;
	std::vector<EvaluatedContour> evaluatedparentsegments; 
    for ( int i = 0; i < detectedcontours.size(); i++ ) {
        if ( detectedhierarchy[i][3] > -1 ) {
			EvaluateSegment( detectedcontours[i], image.rows, evaluatedchildsegments );
        } else {
			EvaluateSegment( detectedcontours[i], image.rows, evaluatedparentsegments );
		}
    }

//-----------------------------------------------------------------------------------------
//Construct from segments
//-----------------------------------------------------------------------------------------	
    std::vector<std::vector<cv::Point>> constructedcontours;
	ConstructFromSegments( evaluatedchildsegments, constructedcontours );
	//ConstructFromSegments( evaluatedparentsegments, constructedcontours );
	//std::cout << "Contours constructed: " << constructedcontours.size() << std::endl;

//-----------------------------------------------------------------------------------------
//Evaluate constructed segments
//-----------------------------------------------------------------------------------------	
	for ( Contour contour : constructedcontours ) {
		EvaluateSegment( contour, image.rows, evaluatedparentsegments );
	}
	
//-----------------------------------------------------------------------------------------
//Filter and sort all evaluated contours
//-----------------------------------------------------------------------------------------	
	std::vector<EvaluatedContour> leftcontours;
	std::vector<EvaluatedContour> rightcontours;
	SortContours( evaluatedparentsegments, image.cols, leftcontours, rightcontours );
	//std::cout << "Left pairs: " << leftcontours.size() << std::endl;
	//SortContours( evaluatedchildsegments, image.cols, leftcontours, rightcontours );
	//std::cout << "Right pairs: " << rightcontours.size() << std::endl;
	
//-----------------------------------------------------------------------------------------
//Find highest scoring pair of contours
//-----------------------------------------------------------------------------------------	
	Polygon bestpolygon{ cv::Point(0,0), cv::Point(0,0), cv::Point(0,0), cv::Point(0,0) };
	float maxscore{lanedetectconstants::klowestscorelimit};
	
	//Create optimal polygon mat
	cv::Mat optimalmat{ cv::Mat(POLYGONSCALING * image.rows, POLYGONSCALING * image.cols, CV_8UC1,
		cv::Scalar(0)) };
	cv::Point cvpointarray[4];
	for  (int i =0; i < 4; i++ ) {
		cvpointarray[i] = cv::Point(POLYGONSCALING * lanedetectconstants::optimalpolygon[i].x,
			POLYGONSCALING * lanedetectconstants::optimalpolygon[i].y);
	}
	cv::fillConvexPoly( optimalmat, cvpointarray, 4,  cv::Scalar(1) );

	//Find best score
	for ( EvaluatedContour &leftevaluatedontour : leftcontours ) {
		for ( EvaluatedContour &rightevaluatedcontour : rightcontours ) {
			Polygon newpolygon{ cv::Point(0,0), cv::Point(0,0), cv::Point(0,0),
				cv::Point(0,0) };
			FindPolygon( newpolygon, leftevaluatedontour.contour,
				rightevaluatedcontour.contour );
			//If invalid polygon created, goto next
			if ( newpolygon[0] == cv::Point(0,0) ) continue;
			//If valid score
			//float score{ ScoreContourPair( newpolygon, image.cols, image.rows,
			//	leftevaluatedontour, rightevaluatedcontour) };
			float score = PercentMatch(newpolygon, optimalmat);
			//float score = PercentMatch2(newpolygon, optimalmat, optimalpolygonarea);
			//float score = PercentMatch3(newpolygon, lanedetectconstants::optimalpolygon,
			//	optimalpolygonarea);
			//If highest score update
			if ( score > maxscore ) {
				maxscore = score;
				bestpolygon = newpolygon;
			}
		}
	}
	
//-----------------------------------------------------------------------------------------
//Return results
//-----------------------------------------------------------------------------------------	
	std::copy(std::begin(bestpolygon), std::end(bestpolygon), std::begin(polygon));
	return;
}

/*****************************************************************************************/	
void EvaluateSegment( const Contour& contour,
                      const int imageheight,
					  std::vector<EvaluatedContour>& evaluatedsegments )
{
	//Performance note: Evaluating ellipse vs fitline first seems about the same
	
	//Filter by size, only to prevent exception when creating ellipse or fitline
	if ( contour.size() < 5 ) return;
		
	//Create ellipse
	cv::RotatedRect ellipse{ fitEllipse(contour) };
	
	//Filter by screen position
	if ( ellipse.center.y < (imageheight * 0.6f)) return;
	
	//Filter by length (ellipse vs segment?)
	if ( ellipse.size.height < lanedetectconstants::ksegmentellipseheight ) return;
	//if ( arcLength(contour, false) < lanedetectconstants::ksegmentlength ) return;
	
	//Calculate length to width ratio
	float lengthwidthratio{ ellipse.size.height / ellipse.size.width };
	
	//Filter by length to width ratio
	if ( lengthwidthratio < lanedetectconstants::ksegmentlengthwidthratio ) return;
		
	//Create fitline
	cv::Vec4f fitline;
	cv::fitLine(contour, fitline, CV_DIST_L2, 0, 0.1, 0.1 );
	
	//Filter by angle
	float angle{ FastArcTan(fitline[1] / fitline[0]) };
	if ( abs(angle - 90.0f) > lanedetectconstants::ksegmentanglewindow ) return;
	//if ( abs(ellipse.angle - 90.0f) > lanedetectconstants::ksegmentanglewindow )

	evaluatedsegments.push_back( EvaluatedContour{contour, ellipse, lengthwidthratio,
		angle, fitline} );
	return;
}

/*****************************************************************************************/	
void ConstructFromSegments( const  std::vector<EvaluatedContour>& evaluatedsegments,
                            std::vector<Contour>& constructedcontours )
{
    for ( const EvaluatedContour &segcontour1 : evaluatedsegments ) {
		for ( const EvaluatedContour &segcontour2 : evaluatedsegments ) {
			float createdangle { FastArcTan((segcontour1.ellipse.center.y -
				segcontour2.ellipse.center.y) / (segcontour1.ellipse.center.x -
			segcontour2.ellipse.center.x)) };
			float angledifference1( abs(segcontour1.angle -	segcontour2.angle) );
			float angledifference2( abs(createdangle -	segcontour1.angle) );
			float angledifference3( abs(createdangle -	segcontour2.angle) );
			if ((angledifference1 < lanedetectconstants::ksegmentsanglewindow) &&
				(angledifference2 < lanedetectconstants::ksegmentsanglewindow) &&
				(angledifference3 < lanedetectconstants::ksegmentsanglewindow)) {
				Contour newcontour{ segcontour1.contour };
				newcontour.insert( newcontour.end(), segcontour2.contour.begin(),
					segcontour2.contour.end() );
				constructedcontours.push_back( newcontour );
			}
		}
    }	
	return;
}

/*****************************************************************************************/
void SortContours( const std::vector<EvaluatedContour>& evaluatedsegments,
                   const int imagewidth,
				   std::vector<EvaluatedContour>& leftcontours,
				   std::vector<EvaluatedContour>& rightcontours )
{
	for ( const EvaluatedContour &evaluatedcontour : evaluatedsegments ) {
		//Filter by length (ellipse vs segment?)
		if ( evaluatedcontour.ellipse.size.height < lanedetectconstants::kellipseheight )
			continue;
		//if ( evaluatedcontour.contour.arcLength(contour, false) <
		//	lanedetectconstants::klength ) continue;
		
		//Filter by angle
		if ( abs(evaluatedcontour.angle - 90.0f) >
			lanedetectconstants::kanglewindow ) continue;
			
		//Filter by length to width ratio
		if ( evaluatedcontour.lengthwidthratio < lanedetectconstants::klengthwidthratio )
			continue;
		
		//Push into either left or right evaluated contour set
		if ( evaluatedcontour.ellipse.center.x < (imagewidth * 0.5f) ) {
			leftcontours.push_back( evaluatedcontour );
		} else {
			rightcontours.push_back( evaluatedcontour );
		}
	}
	return;
}

/*****************************************************************************************/
void FindPolygon( Polygon& polygon,
                  const Contour& leftcontour,
				  const Contour& rightcontour,
				  bool useoptimaly )
{
	//Check for valid contours to prevent exception
	/*
	if ( leftcontour.empty() || rightcontour.empty() ) {
        return;
    }*/
	
	//Get point extremes
	auto minmaxyleft = std::minmax_element(leftcontour.begin(), leftcontour.end(),
		[]( const cv::Point& lhs, const cv::Point& rhs ) { return lhs.y < rhs.y; });
	auto minmaxyright = std::minmax_element(rightcontour.begin(), rightcontour.end(),
		[]( const cv::Point& lhs, const cv::Point& rhs ) { return lhs.y < rhs.y; });
	int leftmaxx{minmaxyleft.second->x}, leftminx{minmaxyleft.first->x},
		leftmaxy{minmaxyleft.second->y}, leftminy{minmaxyleft.first->y};
	int rightmaxx{minmaxyright.second->x}, rightminx{minmaxyright.first->x},
		rightmaxy{minmaxyright.second->y}, rightminy{minmaxyright.first->y};
	int maxy;
	if (useoptimaly) {
		maxy = lanedetectconstants::optimalpolygon[0].y;
	} else {
		maxy = std::max(minmaxyleft.second->y, minmaxyright.second->y);
	}
	int miny{std::max(minmaxyleft.first->y, minmaxyright.first->y)};
	
	//Define slopes
	float leftslope{ static_cast<float>(leftmaxy-leftminy)/static_cast<float>(
		leftmaxx - leftminx) };
    float rightslope{ static_cast<float>(rightmaxy-rightminy)/static_cast<float>(
		rightmaxx - rightminx) };
    cv::Point leftcenter = cv::Point((leftmaxx + leftminx) * 0.5f,(leftmaxy + leftminy) * 0.5f);
    cv::Point rightcenter = cv::Point((rightmaxx + rightminx) * 0.5f,(rightmaxy + rightminy) * 0.5f);

	//If valid slopes found, calculate 4 vertices of the polygon
    if ( (std::fpclassify(leftslope) == FP_NORMAL) && (std::fpclassify(rightslope) == FP_NORMAL) ){
        //Calculate points
        cv::Point bottomleft = cv::Point(leftcenter.x +
			(maxy - leftcenter.y)/leftslope, maxy);
        cv::Point bottomright = cv::Point(rightcenter.x +
			(maxy - rightcenter.y)/rightslope, maxy);
        cv::Point topright = cv::Point(rightcenter.x -
			(rightcenter.y - miny)/rightslope, miny);
        cv::Point topleft = cv::Point(leftcenter.x -
			(leftcenter.y - miny)/leftslope, miny);
        //Check validity of points
        if ((((leftslope < 0.0f) && (rightslope > 0.0f)) ||
            ((leftslope > 0.0f) && (rightslope > 0.0f)) ||
            ((leftslope < 0.0f) && (rightslope < 0.0f))) &&
            ((bottomleft.x < bottomright.x) && (topleft.x < topright.x))){

            //Construct polygon
			polygon[0] = bottomleft;
			polygon[1] = bottomright;
			polygon[2] = topright;
			polygon[3] = topleft;
        }
    }
	return;
}

/*****************************************************************************************/
float ScoreContourPair( const Polygon& polygon,
                        const int imagewidth,
						const int imageheight,
						const EvaluatedContour& leftcontour,
						const EvaluatedContour& rightcontour )
{
	//Filter by common angle
	float deviationangle{ 180.0f - leftcontour.angle -	rightcontour.angle };
	if ( abs(deviationangle) > lanedetectconstants::kcommonanglewindow ) return (-FLT_MAX);
	
	//Filter by road width
	int roadwidth{ polygon[1].x - polygon[0].x };
	if ( roadwidth < lanedetectconstants::kminroadwidth ) return (-FLT_MAX);
	if ( roadwidth > lanedetectconstants::kmaxroadwidth ) return (-FLT_MAX);
	
	//Calculate score
	float weightedscore{ 0.0f };
	weightedscore += lanedetectconstants::kellipseratioweight * (
		leftcontour.lengthwidthratio + rightcontour.lengthwidthratio);
	weightedscore += lanedetectconstants::kangleweight * abs(deviationangle);
	weightedscore += lanedetectconstants::kcenteredweight * (
		abs(imagewidth - polygon[0].x - polygon[1].x));
	weightedscore += lanedetectconstants::kwidthweight * (
		abs(lanedetectconstants::koptimumwidth -(polygon[1].x - polygon[0].x)));
	weightedscore += lanedetectconstants::klowestpointweight * (
		imageheight - polygon[0].y);
	return weightedscore;
}

/*****************************************************************************************/
float PercentMatch( const Polygon& polygon,
					const cv::Mat& optimalmat )
{
	//Create blank mat
	cv::Mat polygonmat{ cv::Mat(optimalmat.rows, optimalmat.cols, CV_8UC1, cv::Scalar(0)) };
	
	//Draw polygon
	cv::Point cvpointarray[4];
	for  (int i =0; i < 4; i++ ) {
		cvpointarray[i] = cv::Point(POLYGONSCALING * polygon[i].x, POLYGONSCALING *
			polygon[i].y);
	}
	cv::fillConvexPoly( polygonmat, cvpointarray, 4,  cv::Scalar(2) );

	//Add together
	polygonmat += optimalmat;
	
	//Evaluate result
	uint16_t excessarea{ 0 };
	uint16_t overlaparea{ 0 };
	for (int i = 0; i < optimalmat.rows; i++) {
		uchar* p { polygonmat.ptr<uchar>(i) };
		for (int j = 0; j < optimalmat.cols; j++) {
			switch ( p[j] )
			{
				case 1:
					excessarea++;
					break;
				case 2:
					excessarea++;
					break;
				case 3:
					overlaparea++;
					break;
			}
		}
	}

	return (100.0f * overlaparea) / (overlaparea + excessarea);
}

/*****************************************************************************************/
float PercentMatch2( const Polygon& polygon,
					 const cv::Mat& optimalmat,
					 const uint16_t optimalarea )
{
	//Create blank mat
	cv::Mat polygonmat{ cv::Mat(optimalmat.rows, optimalmat.cols, CV_8UC1, cv::Scalar(0)) };
	
	//Draw polygon
	cv::Point cvpointarray[4];
	for  (int i =0; i < 4; i++ ) {
		cvpointarray[i] = cv::Point(POLYGONSCALING * polygon[i].x, POLYGONSCALING *
			polygon[i].y);
	}
	cv::fillConvexPoly( polygonmat, cvpointarray, 4,  cv::Scalar(2) );

	//Find area
	uint16_t a( cvpointarray[2].x - cvpointarray[3].x );
	uint16_t b( cvpointarray[1].x - cvpointarray[0].x );
	uint16_t h( cvpointarray[0].y - cvpointarray[2].y );
	uint16_t polygonarea( 0.5f * (a + b) * h );
	
	//Add together
	cv::bitwise_and(polygonmat, optimalmat, polygonmat);
	uint16_t overlaparea { static_cast<uint16_t>(countNonZero(polygonmat)) };

	return (100.0 * overlaparea) / (optimalarea + polygonarea - overlaparea);
}

/*****************************************************************************************/
float PercentMatch3( const Polygon& polygon,
					 const Polygon& optimalpolygon,
					 const uint16_t optimalarea )
{
	//Find area
	uint16_t a( polygon[2].x - polygon[3].x );
	uint16_t b( polygon[1].x - polygon[0].x );
	uint16_t h( polygon[0].y - polygon[2].y );
	uint16_t polygonarea( static_cast<uint16_t>(0.5f * (a + b) * h) );
	
	//Create overlap polygon
	Polygon overlappolygon { cv::Point(0,0), cv::Point(0,0), cv::Point(0,0),
		cv::Point(0,0) };
	//Point 0
	if ( optimalpolygon[0].x > polygon[0].x ) {
		overlappolygon[0].x = optimalpolygon[0].x;
	} else {
		overlappolygon[0].x = polygon[0].x;
	}
	if ( optimalpolygon[0].y < polygon[0].y ) {
		overlappolygon[0].y = optimalpolygon[0].y;
		overlappolygon[1].y = optimalpolygon[0].y;
	} else {
		overlappolygon[0].y = polygon[0].y;
		overlappolygon[1].y = polygon[0].y;
	}
	//Point 1
	if ( optimalpolygon[1].x < polygon[1].x ) {
		overlappolygon[1].x = optimalpolygon[1].x;
	} else {
		overlappolygon[1].x = polygon[1].x;
	}
	//Point 2
	if ( optimalpolygon[2].x < polygon[2].x ) {
		overlappolygon[2].x = optimalpolygon[2].x;
	} else {
		overlappolygon[2].x = polygon[2].x;
	}
	if ( optimalpolygon[2].y > polygon[2].y ) {
		overlappolygon[2].y = optimalpolygon[2].y;
		overlappolygon[3].y = optimalpolygon[2].y;
	} else {
		overlappolygon[2].y = polygon[2].y;
		overlappolygon[3].y = polygon[2].y;
	}
	//Point 3
	if ( optimalpolygon[3].x > polygon[3].x ) {
		overlappolygon[3].x = optimalpolygon[3].x;
	} else {
		overlappolygon[3].x = polygon[3].x;
	}
		
	
	//Find area
	uint16_t aovr( overlappolygon[2].x - overlappolygon[3].x );
	uint16_t bovr( overlappolygon[1].x - overlappolygon[0].x );
	uint16_t hovr( overlappolygon[0].y - overlappolygon[2].y );
	uint16_t overlaparea( static_cast<uint16_t>(0.5f * (aovr + bovr) * hovr) );

	return (100.0 * overlaparea) / (optimalarea + polygonarea - overlaparea);
}

/*****************************************************************************************/
int32_t ScorePolygonByPoint( const Polygon& polygon,
							 const Polygon& optimalpolygon )
{
	int32_t score {0};
	for (int i = 0; i < 4; i++) {
		cv::Point diff { polygon[i] - optimalpolygon[i] };
		//The literal multiplied by y is to emphasise focus on x
		score -= FastSquareRoot((diff.x * diff.x) + 0.75 * (diff.y * diff.y));
	}
	return score;
}

/*****************************************************************************************/
void AveragePolygon ( Polygon& polygon,
                      std::deque<Polygon>& pastpolygons,
					  int samplestoaverage,
					  int samplestokeep )
{
	//FIFO
	pastpolygons.push_back( polygon );
	if ( pastpolygons.size() > samplestokeep ) {
		pastpolygons.pop_front();
	}
	//Sum nonzero
	Polygon averagepolygon { cv::Point(0,0), cv::Point(0,0), cv::Point(0,0),
		cv::Point(0,0) };
	int nonzerocount{0};
	for ( Polygon &ipolygon : pastpolygons ) {
		if ( ipolygon[0] == cv::Point(0,0) ) continue;
		nonzerocount++;
		for (int i = 0; i < ipolygon.size(); i++) {
			averagepolygon[i].x += ipolygon[i].x;
			averagepolygon[i].y += ipolygon[i].y;
		}
	}	
	if ( nonzerocount == 0 ) return;
	//Average nonzero
	for ( int i = 0; i < polygon.size(); i++ ) {
		averagepolygon[i].x /= nonzerocount;
		averagepolygon[i].y /= nonzerocount;
	}
	
	//if not enough nonzero polygons, return
	if ( nonzerocount < samplestoaverage ) {
		std::copy(std::begin(averagepolygon), std::end(averagepolygon),
			std::begin(polygon));
		return;
	}
	//Find differences
	std::vector<PolygonDifferences> polygondifferences;
	for ( Polygon &ipolygon : pastpolygons ) {
		float differencefromaverage{0.0f};
		for (int i = 0; i < ipolygon.size(); i++) {
			differencefromaverage += abs(averagepolygon[i].x - ipolygon[i].x);
			differencefromaverage += abs(averagepolygon[i].y - ipolygon[i].y);
		}
		polygondifferences.push_back( PolygonDifferences { ipolygon,
			differencefromaverage } );
	}
	//Sort
	sort(polygondifferences.begin(), polygondifferences.end(), [](
		const PolygonDifferences& a, const PolygonDifferences& b ) {
		return a.differencefromaverage < b.differencefromaverage; });
	//Sum closest values
	averagepolygon = { cv::Point(0,0), cv::Point(0,0), cv::Point(0,0),
		cv::Point(0,0) };
	for (int i = 0; i < samplestoaverage; i++) {
		for (int j = 0; j < 4; j++) {
			averagepolygon[j].x += polygondifferences[i].polygon[j].x;
			averagepolygon[j].y += polygondifferences[i].polygon[j].y;
		}
	}
	//Average closest values
	for ( int i = 0; i < polygon.size(); i++ ) {
		averagepolygon[i].x /= samplestoaverage;
		averagepolygon[i].y /= samplestoaverage;
	}
	std::copy(std::begin(averagepolygon), std::end(averagepolygon),
		std::begin(polygon));
	return;
}

/*****************************************************************************************/
uint32_t FastSquareRoot( int32_t x )
{
    int32_t a, b;
    b = x;
    a = x = 0x3f;
    x = b / x;
    a = x = (x + a) >> 1;
    x = b / x;
    a = x = (x + a) >> 1;
    x = b / x;
    x = (x + a) >> 1;
    return x;  
}

/*****************************************************************************************/
float FastArcTan( const double slope )
{
    return ( 90.f + DEGREESPERRADIAN * ( M_PI_4 * slope - slope * (fabs(slope) - 1) *
		(0.2447 + 0.0663 * fabs(slope))) );
}
  