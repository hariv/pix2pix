#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/text.hpp"

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::text;




// Used to extract text segments from the gift Card
// need to add aspect ratio
vector<Mat> separateChannels(Mat& src)
{
	vector<Mat> channels;
	// if it is a gray scale image, we will only have two channels, img and its negative
	if(src.type() == CV_8U || src.type() == CV_8UC1)
	{	// if gray scale
		channels.push_back(src); // add the image
		channels.push_back(255 - src); // add the negative of the image to the vector
		return channels;
	}
	
	// colored images // if colored image, then we will have many channels total 6 RBG = -ves
	if(src.type() == CV_8UC3)
	{
		computeNMChannels(src, channels);
		// the above function seprates the image into compnent channels
		// channel is a vector of Mats filled with resultant channels
		int size = static_cast<int>(channels.size()) - 1;
		for(int c = 0; c<size; c++)
			channels.push_back(255 - channels[c]);
		return channels;
		// the above for loops is written to append all the negatives of the color components
		// to the vector. Negatives are appeneded so that the algorithm will cover both
		// bright text on dark background and dark text on light background
	
	}
	//for any other kind of image, the function will terminate
	cout << "Invalid image format!" << endl;
	exit(-1); // if the control reaches here, it is an error.



}

// To remove the skew, see the last program for explantion
// This is called in drawER function , see drawER function 

Mat deskewAndCrop(Mat input, const RotatedRect& box)
{
    double angle = box.angle;
    Size2f size = box.size;
    
    //Adjust the box angle
    if (angle < -45.0)
    {
        angle += 90.0;
        std::swap(size.width, size.height);
    }
    
    //Rotate the text according to the angle
    Mat transform = getRotationMatrix2D(box.center, angle, 1.0);
    Mat rotated;
    warpAffine(input, rotated, transform, input.size(), INTER_CUBIC);
    
    //Crop the result
    Mat cropped;
    getRectSubPix(rotated, size, box.center, cropped);
    copyMakeBorder(cropped,cropped,10,10,10,10,BORDER_CONSTANT,Scalar(0));
    return cropped;
}


// The following function is called after separateChannels and after calling NMfilters to
// detect text regions in the main method -- see the main method.
// Now that we have detected the regions, must crop out the text before submiting to ocr
// we can use input(rect) and draw all regions or send thse to ocr but if the letter are
// skewed some relavant tet may get cropped out.
// So we use the ERStat object from ERFilter that gives use pixels inside each extremal region
// With these pixels we use the floodfill function to redraw or reconstruct each letter.
// this function will paint similar colored pixels based on seed points

Mat drawER(const vector<Mat> &channels, const vector<vector<ERStat> > &regions, 
								const vector<Vec2i>& group, const Rect& rect)
// this function takes the vector of all processed channels, ERStat regions,
// A group that must be drawn, note called for each group
// and a group rectangle.
{
    Mat out = Mat::zeros(channels[0].rows+2, channels[0].cols+2, CV_8UC1); // channels[0] since each channel has same number of rows etc.
    // out is the mask that needs to be passed to the floodfill function
    // the mask is the image with 2 rows and 2 columns greater than the input image.
    // floodfill function will draw on the mask and not the input image.
    // when flood fill draws a pixel it verifies that the mask pixel correponding to it
    // is zero, if not it won't draw, that is why mask is zeros.
    
    //flags is a bit mask, 4 means all 4 edge pixels are used, in case of 8, diagonal pixels are used as wel
    
    int flags = 4                    //4 neighbors,  least significant 8 bits
    + (255 << 8)                //paint mask in white (255), next significant 8 bits, 255 means white color used to fill the mask
    + FLOODFILL_FIXED_RANGE        //fixed range
    + FLOODFILL_MASK_ONLY;        //Paint just the mask, not the image
    
    for (int g=0; g < group.size(); g++)
    {  // Now we need to loop through each group, since we need to find the region index
    	// and stats such that the region is not root which contains no points, ignore it
    
    
        int idx = group[g][0]; // gth extremal region for 1st word
        ERStat er = regions[idx][group[g][1]];
        
        //Ignore root region
        if (er.parent == NULL)
            continue;
            
        //Now, we can read the pixel coordinate from the ERStat object. It's represented by
//the pixel number, counting from top to bottom, left to right. This linear index must
//be converted to a row (y) and column (z) notation
        
        //Transform the linear pixel value to row and col
        int px = er.pixel % channels[idx].cols;
        int py = er.pixel / channels[idx].cols;
        
        //Create the point and adds it to the list.
        Point p(px, py); // seed or starting point
        
        //Draw the extremal region
        floodFill(
                  channels[idx], out,                //Image and mask
                  p, Scalar(255),                    //Seed and color
                  nullptr,                        //No rect
                  Scalar(er.level),Scalar(0),        //LoDiff and upDiff
                  flags                            //Flags
                  );
    }
    
    // After we have done it for all regions, we will end up with an image a little bigger
    // with black background and word in white letters now lets us just crop out area of the 
    // letters since region rectangle is given, so we start by defining it as our region  
    // of interest
    
    //Crop just the text area and find it's points
    out = out(rect);
    
    //then find all non zero pixels, which we will use in minAreaRect to get the rotated 
    //rectangle around the letters and then use deskew function to crop and rotate the image
    // for us.
    
    vector<Point> points;
    findNonZero(out, points);
    //Use deskew and crop to crop it perfectly
    return deskewAndCrop(out, minAreaRect(points));
}



tesseract::TessBaseAPI ocr;
// we create a global TessBaseAPI object that represents our Tesseract OCR engine

// Now we create a function that will run the OCR

char* identifyText(Mat input, char* language = "eng") // input is the input image
{// eng will load english, por will load portuguese, "eng+por" will load both
	//ocr.Init(NULL, language, tesseract::OEM_TESSERACT_CUBE_COMBINED);
	ocr.Init(NULL, language, tesseract::OEM_TESSERACT_ONLY);
	// for Init the first parameter is the datapath, the path to the tessdata files of the
	// root directory. NULL will it search in the installation directory. When deployed
	// this argument is furnished in argv.
	// last parameter is the OCR algorithm
	// we use the one with maximum accuracy, can also use OEM_TESSERACT_ONLY
	// we can do init multiple times to load an OCR engine for a different language
	// can also do multiple ocr's in multiple threads using initialized objects
	ocr.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	// This sets the page segmentation mode. Some examples are PSM_OSD_ONLY - here it will
	// just run its preprocessing algorithm to detect the orientation and script detection
	// PSM_AUTO_OSD does auto page segmentation with orientation and script detection.
//PSM_SINGLE_LINE indicates that image contains only one line of text
//PSM_SINGLE_WORD says that the image contains just one word
//We use PSM_SINGLE_BLOCK since the preprocessin done earlier in last exercise guarentees that

	// Tesseract does its own preprocessing but it is important to know how to do it for our own
	//custom applications
	ocr.SetImage(input.data, input.cols, input.rows, 1, input.step); // input.step is the no. of bytes per line
	// now let us run the recongition and return it as encoded UTF8
	char* text = ocr.GetUTF8Text();
	cout << "Text:" << endl;
	cout << text << endl;
	cout << "Confidence: " << ocr.MeanTextConf() << endl << endl;
	// MeanTextConf return a confidence index from 0 to 100
	// get text
	return text;
	
}


int main(int argc, char** argv)
{
	auto input = imread(argv[1]);
	//Mat processed;
	Mat processed = input;
	//cvtColor(input, processed, CV_RGB2GRAY); // convert the image to gray scale
	auto channels = separateChannels(processed); // and separate channels
	
	// we may also want to work with a colored image and not convert to gray scale
	// in that case uncomment the following line
	// Mat processed = input;
	// also the middle two lines the the above block of 4 lines
	// we will have 6 channels RGB + inveterd RGB
	
	//............................................................................
	//Now as we said before, the classifier works in two stages
	// create ERFilter objects with 1st and 2nd stage classifiers
	auto filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 15, 
								0.00015f, 0.13f, 0.2f, true, 0.1f);
				
	auto filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5);	
	
	// For filter1, the arguments are 
	// the first argument is the const pointer to the classification model, this is returned by loadClassifier() method
	// second arg is threshold delta, the amount to be added to threshold each time the algotithm runs
	// //third arg is the minArea of ER where text can be found, it is % of image size, ER smaller than this area are discarded.
	//4th arg is the maxArea of ER where text can be found, also % of image size.
	//5th arg is the minProbability that a regions must have to be a character in order for the region to remain for next stage.
	//6th arg is wheter nonMax suppresion is done in each branch probablity 
	//7th arg is th minimum probility difference between the minim and max extreme region
	
	//for filter2 , the function only takes the ptr to model and min prob that a region must achieve
	//in order to be considered a character.
	
	
	//Now we call the algorithm in each channel to identify all possible text regions:
	
	//Extract text regions using NM algorithm
	cout << "Processing" << channels.size() << "Channels...." ;
	cout << endl;
	
	vector<vector<ERStat> > regions(channels.size()); // declaring data structur to be filled with
	
	for (int c = 0; c < channels.size(); c++)
	{
		cout << "  Channel " << (c+1) << endl;
		filter1->run(channels[c], regions[c]);
		filter2->run(channels[c], regions[c]);
		// run function above takes two args, the first argument is the input or image chanel to be processed
		// the second arg in the filter1 will be filled with detected region, in second stage
		//filter 2, this argument must contain regions selected by filter1 which wil be processed
		// and filtered in stage2
	}
	filter1.release();
	filter2.release(); // release filters
	
	// now we need to group all ERRegions into possible words and define their bounding boxes
	// by calling erGrouping function
	
	// Separate character groups from regions
	vector< vector<Vec2i> > groups;
	vector<Rect> groupRects;
	erGrouping(input, channels, regions, groups, groupRects, ERGROUPING_ORIENTATION_HORIZ);
	
	cout << groupRects[0] << endl;
	
	//input is the image to be processed
	//regions is the vector of single channel images where regions are extracted
	//groups is the output vector of indexes of grouped regions. Each group region contains
	// all extremal regions of a single word.
	
	//groupRects : is the list of rectangles with the detected text regions
	// last is the method, our method will only detect horizontally oriented text ,
	// we can also do any orientation text, but we need to pass in a classifier for that
	
	//erGrouping(input, channels, regions, groups, groupRects, ERGROUPING_ORIENTATION_ANY, 
	//						"trained_classifier_erGrouping.xml", 0.5);
	
	// the above call will do any orientation
	
	//.....................................
	//Finally draw groups boxes
	
	for(auto rect : groupRects)
	{
		rectangle(input, rect, Scalar(0,255,0), 3); //draw green rect of thickenss 3
		//Mat sub = input(rect);  // code to extract individual parts
		//imshow("New REgions", sub);
		//waitKey(5000);
	}
	imshow("grouping", input);
	waitKey(5000);
	
	stringstream ss;
    int z = 1;
	for (int i = 0; i < groups.size(); i++) 
	{
		 Mat wordImage = drawER(channels, regions, groups[i], groupRects[i]);
		 // now call the ocr engine
		 identifyText(wordImage);
		 imshow("REgion", wordImage);
         ss << "RegionGiftCard " << z <<".png";
         //writes the regions on interest as images
         imwrite(ss.str(), wordImage);
         ss.str(std::string()); // clear the string stream variable
         z++;
		 
		 waitKey(1000);
	}
	
	
	
	
		
	
	
	
	
	
	
	
			
				
				
				


	return 0;

}
