﻿// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>
#include <unordered_map>

using namespace std;
wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

//ex1
template <typename T>
constexpr T clamp(T value, T min, T max) {
	return (value < min) ? min : (value > max) ? max : value;
}

void processImage() {
	char filePath[MAX_PATH];
	while (openFileDlg(filePath)) {
		Mat inputImage = imread(filePath, IMREAD_GRAYSCALE);
		if (inputImage.empty()) {
			cout << "Error: Image not found!" << endl;
			return;
		}

		Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC3, Scalar(255, 255, 255));
		double shapeArea = 0, centerX = 0, centerY = 0;
		double sumNumerator = 0, sumDenominator = 0, shapePerimeter = 0;

		Vec3b blueColor(255, 0, 0), blackColor(0, 0, 0);

		for (int row = 0; row < inputImage.rows; row++) {
			for (int col = 0; col < inputImage.cols; col++) {
				if (inputImage.at<uchar>(row, col) == 0) {
					shapeArea++;
					outputImage.at<Vec3b>(row, col) = blueColor;
					centerX += row;
					centerY += col;
				}
			}
		}

		centerX /= shapeArea;
		centerY /= shapeArea;

		for (int offset = -2; offset <= 2; offset++) {
			outputImage.at<Vec3b>(clamp((int)centerX + offset, 0, inputImage.rows - 1), (int)centerY) = blackColor;
			outputImage.at<Vec3b>((int)centerX, clamp((int)centerY + offset, 0, inputImage.cols - 1)) = blackColor;
		}

		for (int row = 0; row < inputImage.rows; row++) {
			for (int col = 0; col < inputImage.cols; col++) {
				if (inputImage.at<uchar>(row, col) == 0) {
					sumNumerator += 2 * (row - centerX) * (col - centerY);
					sumDenominator += pow(col - centerY, 2) - pow(row - centerX, 2);
				}
			}
		}
		double orientationAngle = atan2(sumNumerator, sumDenominator) / 2;
		double angleDegrees = orientationAngle * 180 / CV_PI;

		line(outputImage, Point(centerY + 200 * cos(orientationAngle), centerX + 200 * sin(orientationAngle)),
			Point(centerY - 200 * cos(orientationAngle), centerX - 200 * sin(orientationAngle)), Scalar(0, 0, 255));

		for (int row = 1; row < inputImage.rows - 1; row++) {
			for (int col = 1; col < inputImage.cols - 1; col++) {
				if (inputImage.at<uchar>(row, col) == 0) {
					if (inputImage.at<uchar>(row + 1, col) != 0 || inputImage.at<uchar>(row - 1, col) != 0 ||
						inputImage.at<uchar>(row, col + 1) != 0 || inputImage.at<uchar>(row, col - 1) != 0) {
						shapePerimeter++;
						outputImage.at<Vec3b>(row, col) = blackColor;
					}
				}
			}
		}

		double compactness = 4 * CV_PI * (shapeArea / (shapePerimeter * shapePerimeter));

		int minRow = inputImage.rows, maxRow = 0, minCol = inputImage.cols, maxCol = 0;
		for (int row = 0; row < inputImage.rows; row++) {
			for (int col = 0; col < inputImage.cols; col++) {
				if (inputImage.at<uchar>(row, col) == 0) {
					minRow = min(minRow, row);
					maxRow = max(maxRow, row);
					minCol = min(minCol, col);
					maxCol = max(maxCol, col);
				}
			}
		}

		double aspectRatio = (maxCol - minCol + 1.0) / (maxRow - minRow + 1.0);
		rectangle(outputImage, Point(minCol, minRow), Point(maxCol, maxRow), Scalar(0, 0, 0), 1);

		cout << "Area = " << shapeArea << endl;
		cout << "Center of mass: row = " << centerX << ", col = " << centerY << endl;
		cout << "Angle: radians = " << orientationAngle << ", degrees = " << angleDegrees << endl;
		cout << "Compactness = " << compactness << endl;
		cout << "Aspect ratio = " << aspectRatio << endl;

		imshow("Original", inputImage);
		imshow("Processed", outputImage);
		waitKey(0);
	}
}

//ex2

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> twoPassLabeling(Mat img) {
	int rows = img.rows, cols = img.cols;
	std::vector<std::vector<int>> labels(rows, std::vector<int>(cols, 0));

	int di[] = { 0, -1, -1, -1 };
	int dj[] = { -1, -1, 0, 1 };

	int label = 0;
	std::vector<std::vector<int>> edges;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (img.at<uchar>(i, j) == 0 && labels[i][j] == 0) {
				std::vector<std::pair<int, int>> L;

				for (int k = 0; k < 4; k++) {
					int ni = i + di[k], nj = j + dj[k];
					if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && labels[ni][nj] > 0) {
						L.push_back({ ni, nj });
					}
				}

				if (L.empty()) {
					label++;
					labels[i][j] = label;
					edges.resize(label + 1);
				}
				else {
					int x = labels[L[0].first][L[0].second];
					for (auto& y : L) {
						x = min(x, labels[y.first][y.second]);
					}

					labels[i][j] = x;
					for (auto& y : L) {
						if (labels[y.first][y.second] != x) {
							edges[labels[y.first][y.second]].push_back(x);
							edges[x].push_back(labels[y.first][y.second]);
						}
					}
				}
			}
		}
	}

	std::vector<std::vector<int>> interLabels = labels;
	int newLabel = 0;
	std::vector<int> newLabels(label + 1, 0);

	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);

			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			labels[i][j] = newLabels[labels[i][j]];
		}
	}

	return { interLabels, labels };
}

void generateColorImageTwoPasses() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		if (img.empty()) {
			cerr << "Error: Could not open the image!" << endl;
			return;
		}

		Mat newImg1(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
		Mat newImg2(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

		std::vector<std::vector<int>> interLabels, labels;
		std::tie(interLabels, labels) = twoPassLabeling(img);

		unordered_map<int, Vec3b> colors1, colors2;
		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<int> dist(0, 255);

		auto getRandomColor = [&]() -> Vec3b {
			return Vec3b(dist(gen), dist(gen), dist(gen));
		};

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (labels[i][j] != 0) {
					if (colors1.find(labels[i][j]) == colors1.end()) {
						colors1[labels[i][j]] = getRandomColor();
					}
					newImg1.at<Vec3b>(i, j) = colors1[labels[i][j]];
				}

				if (interLabels[i][j] != 0) {
					if (colors2.find(interLabels[i][j]) == colors2.end()) {
						colors2[interLabels[i][j]] = getRandomColor();
					}
					newImg2.at<Vec3b>(i, j) = colors2[interLabels[i][j]];
				}
			}
		}

		imshow("Original Image", img);
		imshow("Intermediate Labels", newImg2);
		imshow("Final Labels", newImg1);
		waitKey(0);
	}
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");

		printf(" 51 - Lab 5 ex 1\n");
		printf(" 52 - Lab 5 ex 2\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;

			case 51:
				processImage();
				break;
			case 52:
				generateColorImageTwoPasses();
				break;
		}
	}
	while (op!=0);
	return 0;
}