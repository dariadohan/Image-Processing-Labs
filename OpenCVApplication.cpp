// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#ifndef PI
#define PI 3.141592653589793
#endif

#define CV_LOAD_IMAGE_GRAYSCALE 700
wchar_t* projectPath;
#define CV_LOAD_IMAGE_GRAYSCALE 700


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

void grey_fa() {
	Mat img = imread("Images/cameraman.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = img.at<uchar>(i, j) + 120;
			if (img.at<uchar>(i, j) > 255)
				img.at<uchar>(i, j) = 255;
		}
	}
	imshow("gray levels added", img);
	waitKey(0);
}

void grey_fm() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = img.at<uchar>(i, j) * 2;
			if (img.at<uchar>(i, j) > 255)
				img.at<uchar>(i, j) = 255;
		}
	}
	imshow("gray levels multiplied", img);
	waitKey(0);
}

void create_colored_image() {
	Mat img(256, 256, CV_8UC3, Scalar(255, 255, 255));
	rectangle(img, Point(128, 0), Point(255, 127), Scalar(0, 0, 255), FILLED);  // Roșu
	rectangle(img, Point(0, 128), Point(127, 255), Scalar(0, 255, 0), FILLED);  // Verde
	rectangle(img, Point(128, 128), Point(255, 255), Scalar(0, 255, 255), FILLED); // Galben

	imshow("Colored Quadrants", img);
	waitKey(0);
}

void invert_matrix() {
	Mat mat = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 10);
	Mat mat_inv;
	invert(mat, mat_inv);

	cout << "Matricea inițială:\n" << mat << endl;
	cout << "Matricea inversată:\n" << mat_inv << endl;
	imshow("hjsjssk", mat);
}

void canaleRGB()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_COLOR);
		if (img.empty())
		{
			cout << "Imaginea nu a fost incarcata corect!" << endl;
			continue;
		}

		int rows = img.rows;
		int cols = img.cols;

		Mat blue = Mat::zeros(rows, cols, CV_8UC3);
		Mat green = Mat::zeros(rows, cols, CV_8UC3);
		Mat red = Mat::zeros(rows, cols, CV_8UC3);

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				Vec3b pixel = img.at<Vec3b>(i, j);

				blue.at<Vec3b>(i, j) = Vec3b(pixel[0], 0, 0);
				green.at<Vec3b>(i, j) = Vec3b(0, pixel[1], 0);
				red.at<Vec3b>(i, j) = Vec3b(0, 0, pixel[2]);
			}
		}

		imshow("Blue Channel", blue);
		imshow("Green Channel", green);
		imshow("Red Channel", red);
		waitKey(0);
	}
}

void grayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_COLOR);
		if (img.empty())
		{
			cout << "Imaginea nu a fost incarcata corect!" << endl;
			continue;
		}

		Mat imggray;
		cvtColor(img, imggray, COLOR_BGR2GRAY);

		imshow("Poza netransformata", img);
		imshow("Grayscale", imggray);
		waitKey(0);
	}
}


void binar()
{
	char fname[MAX_PATH];
	int prag;
	cout << "Introduceti valoarea pragului: ";
	cin >> prag;

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		if (img.empty())
		{
			cout << "Imaginea nu a fost incarcata corect!" << endl;
			continue;
		}

		Mat dst;
		threshold(img, dst, prag, 255, THRESH_BINARY);

		imshow("inainte", img);
		imshow("dupa", dst);
		waitKey(0);
	}
}

void convHSV() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		if (img.empty()) {
			cout << "Imaginea nu a fost încărcată corect!" << endl;
			return;
		}

		Mat H(img.rows, img.cols, CV_8UC1);
		Mat S(img.rows, img.cols, CV_8UC1);
		Mat V(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				Vec3b pixel = img.at<Vec3b>(i, j);
				float r = pixel[2] / 255.0f;
				float g = pixel[1] / 255.0f;
				float b = pixel[0] / 255.0f;

				float M = max( r,max( g, b ));
				float m = min( r,min (g, b ));
				float C = M - m;

				float H_val = 0.0f, S_val = 0.0f, V_val = M;

				if (V_val != 0)
					S_val = C / V_val;

				if (C != 0) {
					if (M == r)
						H_val = 60 * (g - b) / C;
					else if (M == g)
						H_val = 120 + 60 * (b - r) / C;
					else if (M == b)
						H_val = 240 + 60 * (r - g) / C;
				}

				if (H_val < 0)
					H_val += 360;

				H.at<uchar>(i, j) = static_cast<uchar>(H_val * 255/360);
				S.at<uchar>(i, j) = static_cast<uchar>(S_val * 255);
				V.at<uchar>(i, j) = static_cast<uchar>(V_val * 255);
			}
		}

		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		waitKey(0);
	}
}

void hist()
{
	char fname[MAX_PATH];
	int h[256];
	int i = 0;
	int j = 0;
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		if (img.empty())
		{
			cerr << "Eroare la deschiderea imaginii: " << fname << endl;
			continue;
		}
		int rows = img.rows;
		int cols = img.cols;
		std::fill_n(h, 256, 0);
		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < cols; j++)
			{
				h[img(i, j)]++;
			}
		}
		imshow("imagine", img);
		showHistogram("histogram", h, 255, 200);
		waitKey(0);
	}
}

void FDP()
{

	char fname[MAX_PATH];
	int h[256];
	int fdp[256];
	int i = 0;
	int j = 0;
	uchar v;

	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		int rows = img.rows;
		int cols = img.cols;
		int dim = rows * cols;

		for (i = 0; i < 256; i++)
		{
			h[i] = 0;
		}

		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < cols; j++)
			{
				v = img.at<uchar>(i, j);
				h[v]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			fdp[i] = (float)h[i] / dim;
			std::cout << h[i] << " ";
		}

	}

}


void showHistogram() {
	Mat image = imread("C:\\Users\\daria\\OneDrive\\Pictures\\Imagini\\westconcordorthophoto.bmp", IMREAD_GRAYSCALE);
	if (image.empty()) {
		std::cerr << "Error: Could not open or find the image!" << std::endl;
		return;
	}
	const std::string name = "Histogram";
	const int hist_cols = 256;
	const int hist_height = 400;
	int hist[hist_cols] = { 0 };
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int pixelValue = image.at<uchar>(y, x);
			hist[pixelValue]++;
		}
	}
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++) {
		if (hist[i] > max_hist) {
			max_hist = hist[i];
		}
	}
	double scale = (max_hist > 0) ? (double)hist_height / max_hist : 0;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1(x, baseline);
		Point p2(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255));
	}
	imshow(name, imgHist);
	waitKey(0);
}

void histNrRedAcc() {
	const int acc = 256;
	char fname[MAX_PATH];
	int* histogram = new int[acc]();
	uchar pixelValue;
	while (openFileDlg(fname)) {
		Mat_<uchar> image = imread(fname, IMREAD_GRAYSCALE);
		if (image.empty()) {
			std::cerr << "Error: Could not open or find the image!" << std::endl;
			continue;
		}
		int rows = image.rows;
		int cols = image.cols;
		int binSize = 256 / acc;
		std::fill(histogram, histogram + acc, 0);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				pixelValue = image.at<uchar>(i, j);
				histogram[pixelValue / binSize]++;
			}
		}
		imshow("Image", image);
		showHistogram("Histogram", histogram, acc, 200);
		waitKey(0);
	}
	delete[] histogram;
}

void praguriMultiple() {
	char fname[MAX_PATH];
	std::vector<int> peaks;
	const int histogramSize = 256;
	const int windowHalfSize = 5;
	const float threshold = 0.0003f;
	int* histogram = new int[histogramSize]();
	float* frequencyDensity = new float[histogramSize]();
	int* quantizedHistogram = new int[histogramSize]();

	if (!openFileDlg(fname)) {
		return;
	}

	Mat img = imread(fname, IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Error: Could not open or find the image!" << std::endl;
		delete[] histogram;
		delete[] frequencyDensity;
		delete[] quantizedHistogram;
		return;
	}

	int rows = img.rows;
	int cols = img.cols;
	std::fill(histogram, histogram + histogramSize, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			uchar pixelValue = img.at<uchar>(i, j);
			histogram[pixelValue]++;
		}
	}

	for (int i = 0; i < histogramSize; i++) {
		frequencyDensity[i] = static_cast<float>(histogram[i]) / (rows * cols);
	}

	peaks.push_back(0);
	for (int k = windowHalfSize; k < histogramSize - windowHalfSize; k++) {
		float sum = 0.0f;
		bool isPeak = true;

		for (int i = k - windowHalfSize; i <= k + windowHalfSize; i++) {
			sum += frequencyDensity[i];
			if (i != k && frequencyDensity[i] > frequencyDensity[k]) {
				isPeak = false;
			}
		}

		float average = sum / (2 * windowHalfSize + 1);
		if (frequencyDensity[k] > average + threshold && isPeak) {
			peaks.push_back(k);
		}
	}

	peaks.push_back(histogramSize - 1);
	std::fill(quantizedHistogram, quantizedHistogram + histogramSize, 0);

	for (int peak : peaks) {
		quantizedHistogram[peak] = histogram[peak];
	}

	showHistogram("Quantized Image Histogram", quantizedHistogram, histogramSize, 256);
	imshow("Image", img);
	waitKey(0);

	delete[] histogram;
	delete[] frequencyDensity;
	delete[] quantizedHistogram;
}

void applyFloydSteinbergErrorDiffusion() {
	char fname[MAX_PATH];
	std::vector<int> peaks;
	const int histogramSize = 256;
	const int windowHalfSize = 5;
	const float threshold = 0.0003f;
	int* histogram = new int[histogramSize]();
	float* frequencyDensity = new float[histogramSize]();
	int* quantizedHistogram = new int[histogramSize]();

	if (!openFileDlg(fname)) {
		return;
	}

	Mat img = imread(fname, IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Error: Could not open or find the image!" << std::endl;
		delete[] histogram;
		delete[] frequencyDensity;
		delete[] quantizedHistogram;
		return;
	}

	int rows = img.rows;
	int cols = img.cols;
	std::fill(histogram, histogram + histogramSize, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			uchar pixelValue = img.at<uchar>(i, j);
			histogram[pixelValue]++;
		}
	}

	for (int i = 0; i < histogramSize; i++) {
		frequencyDensity[i] = static_cast<float>(histogram[i]) / (rows * cols);
	}

	peaks.clear();
	peaks.push_back(0);
	for (int k = windowHalfSize; k < histogramSize - windowHalfSize; k++) {
		float sum = 0.0f;
		bool isPeak = true;

		for (int i = k - windowHalfSize; i <= k + windowHalfSize; i++) {
			sum += frequencyDensity[i];
			if (i != k && frequencyDensity[i] > frequencyDensity[k]) {
				isPeak = false;
			}
		}

		float average = sum / (2 * windowHalfSize + 1);
		if (frequencyDensity[k] > average + threshold && isPeak) {
			peaks.push_back(k);
		}
	}

	peaks.push_back(histogramSize - 1);
	std::fill(quantizedHistogram, quantizedHistogram + histogramSize, 0);

	for (int peak : peaks) {
		quantizedHistogram[peak] = histogram[peak];
	}

	imshow("Image", img);
	showHistogram("Quantized Image Histogram", quantizedHistogram, histogramSize, 256);
	waitKey(0);

	delete[] histogram;
	delete[] frequencyDensity;
	delete[] quantizedHistogram;
}


int computeRegionArea(const Mat& image, int x, int y) {
	Vec3b colorAtPoint = image.at<Vec3b>(y, x);
	return countNonZero((image == colorAtPoint));
}

pair<int, int> findCenterOfMass(const Mat& image, int x, int y) {
	Vec3b colorAtPoint = image.at<Vec3b>(y, x);
	int totalX = 0, totalY = 0, pixelCount = 0;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			if (image.at<Vec3b>(row, col) == colorAtPoint) {
				totalX += col;
				totalY += row;
				pixelCount++;
			}
		}
	}
	return pixelCount ? make_pair(totalY / pixelCount, totalX / pixelCount) : make_pair(-1, -1);
}

double determineElongationAxis(const Mat& image, int x, int y) {
	Vec3b colorAtPoint = image.at<Vec3b>(y, x);
	auto center = findCenterOfMass(image, x, y);
	int factor1 = 0, factor2 = 0, factor3 = 0;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			if (image.at<Vec3b>(row, col) == colorAtPoint) {
				int deltaY = row - center.first;
				int deltaX = col - center.second;
				factor1 += deltaY * deltaX;
				factor2 += deltaX * deltaX;
				factor3 += deltaY * deltaY;
			}
		}
	}
	return atan2(2 * factor1, factor2 - factor3) * 0.5;
}

int computeBoundaryLength(const Mat& image, int x, int y) {
	Vec3b colorAtPoint = image.at<Vec3b>(y, x);
	int boundaryLength = 0;
	for (int row = 1; row < image.rows - 1; row++) {
		for (int col = 1; col < image.cols - 1; col++) {
			if (image.at<Vec3b>(row, col) == colorAtPoint) {
				if (image.at<Vec3b>(row - 1, col) != colorAtPoint || image.at<Vec3b>(row + 1, col) != colorAtPoint ||
					image.at<Vec3b>(row, col - 1) != colorAtPoint || image.at<Vec3b>(row, col + 1) != colorAtPoint) {
					boundaryLength++;
				}
			}
		}
	}
	return boundaryLength;
}

double calculateShapeThinness(const Mat& image, int x, int y) {
	int regionArea = computeRegionArea(image, x, y);
	int boundaryLength = computeBoundaryLength(image, x, y);
	return (4.0 * PI * regionArea) / (boundaryLength * boundaryLength);
}

double computeShapeAspect(const Mat& image, int x, int y) {
	Vec3b colorAtPoint = image.at<Vec3b>(y, x);
	int rowMin = image.rows, rowMax = 0, colMin = image.cols, colMax = 0;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			if (image.at<Vec3b>(row, col) == colorAtPoint) {
				rowMin = min(rowMin, row);
				rowMax = max(rowMax, row);
				colMin = min(colMin, col);
				colMax = max(colMax, col);
			}
		}
	}
	return (double)(colMax - colMin + 1) / (rowMax - rowMin + 1);
}

void visualizeSelectedRegion(Mat& image, int x, int y) {
	Mat modifiedImage = image.clone();
	Vec3b colorAtPoint = image.at<Vec3b>(y, x);
	auto center = findCenterOfMass(image, x, y);
	double axisAngle = determineElongationAxis(image, x, y);
	int lineLength = 50;
	Point point1(center.second - lineLength * cos(axisAngle), center.first - lineLength * sin(axisAngle));
	Point point2(center.second + lineLength * cos(axisAngle), center.first + lineLength * sin(axisAngle));
	circle(modifiedImage, Point(center.second, center.first), 3, Scalar(0, 255, 0), -1);
	line(modifiedImage, point1, point2, Scalar(255, 0, 0), 2);
	imshow("Selected Region", modifiedImage);
}

void mouseEventCallback(int event, int x, int y, int, void* userData) {
	Mat* image = reinterpret_cast<Mat*>(userData);
	if (event == EVENT_LBUTTONDOWN) {
		cout << "Region Area: " << computeRegionArea(*image, x, y) << endl;
		auto center = findCenterOfMass(*image, x, y);
		cout << "Center of Mass: " << center.first << " " << center.second << endl;
		cout << "Elongation Axis: " << determineElongationAxis(*image, x, y) << endl;
		cout << "Boundary Length: " << computeBoundaryLength(*image, x, y) << endl;
		cout << "Thinness Index: " << calculateShapeThinness(*image, x, y) << endl;
		cout << "Shape Aspect: " << computeShapeAspect(*image, x, y) << endl;
		visualizeSelectedRegion(*image, x, y);
	}
}

void analyzeRegionProperties() {
	Mat imageSource;
	char filePath[MAX_PATH];
	while (openFileDlg(filePath)) {
		imageSource = imread(filePath);
		namedWindow("Image Display", 1);
		setMouseCallback("Image Display", mouseEventCallback, &imageSource);
		imshow("Image Display", imageSource);
		waitKey(0);
	}
}

int encodeColor(const Vec3b& color) {
	return (color[0] << 16) | (color[1] << 8) | (color[2]);
}
Mat refineImage(Mat img, int threshold, float angleMin, float angleMax) {
	unordered_map<int, int> pixelCount, sumXY, sumXX, sumYY;
	unordered_map<int, float> rowCenter, colCenter;
	unordered_map<int, float> orientation;
	Mat output(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			Vec3b color = img.at<Vec3b>(y, x);
			int key = encodeColor(color);
			if (key) {
				pixelCount[key]++;
				rowCenter[key] += y;
				colCenter[key] += x;
			}
		}
	}
	for (auto& entry : pixelCount) {
		rowCenter[entry.first] /= static_cast<float>(entry.second);
		colCenter[entry.first] /= static_cast<float>(entry.second);
	}
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			Vec3b color = img.at<Vec3b>(y, x);
			int key = encodeColor(color);
			if (key) {
				sumXY[key] += (y - rowCenter[key]) * (x - colCenter[key]);
				sumXX[key] += (x - colCenter[key]) * (x - colCenter[key]);
				sumYY[key] += (y - rowCenter[key]) * (y - rowCenter[key]);
			}
		}
	}
	for (auto& entry : pixelCount) {
		orientation[entry.first] = atan2(2.0f * sumXY[entry.first], (sumXX[entry.first] - sumYY[entry.first])) * 0.5f;
		if (orientation[entry.first] < 0) orientation[entry.first] += PI;
	}
	int lineLength = 200;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			Vec3b color = img.at<Vec3b>(y, x);
			int key = encodeColor(color);
			if (key && pixelCount[key] < threshold && orientation[key] >= angleMin && orientation[key] <= angleMax) {
				output.at<Vec3b>(y, x) = color;
			}
		}
	}
	for (auto& entry : orientation) {
		int key = entry.first;
		if (pixelCount[key] < threshold && orientation[key] >= angleMin && orientation[key] <= angleMax) {
			Point start, end;
			start.x = colCenter[key] - lineLength * cos(orientation[key]);
			start.y = rowCenter[key] - lineLength * sin(orientation[key]);
			end.x = colCenter[key] + lineLength * cos(orientation[key]);
			end.y = rowCenter[key] + lineLength * sin(orientation[key]);
			line(output, start, end, Scalar(0, 0, 0), 2);
		}
	}
	return output;
}
void analyzeAndDisplay() {
	Mat src;
	char filename[MAX_PATH];
	while (openFileDlg(filename)) {
		src = imread(filename);
		if (src.empty()) {
			cerr << "Error: Could not load image " << filename << endl;
			continue;
		}
		int threshold;
		float angleMin, angleMax;
		printf("\nThreshold: ");
		scanf("%d", &threshold);
		printf("\nAngle Min: ");
		scanf("%f", &angleMin);
		printf("\nAngle Max: ");
		scanf("%f", &angleMax);
		Mat processedImg = refineImage(src, threshold, angleMin, angleMax);
		namedWindow("Refined Image", WINDOW_AUTOSIZE);
		imshow("Refined Image", processedImg);
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

		printf(" 13 - Ex3\n");
		printf(" 14 - Ex4\n");
		printf(" 15 - Ex5\n");
		printf(" 16 - Ex6\n");

		printf("21 - Ex1\n");
		printf("22 - Ex2\n");
		printf("23 - Ex3\n");
		printf("24 - Ex4\n");

//		printf("31 - Ex1\n");
	//	printf("32 - Ex2\n");
		//printf("33 - Ex3\n");
		printf("34 - Ex4\n");
		printf("35 - Ex5\n");
		printf("36 - Ex6\n");

		printf("41 - Ex1\n");
		printf("42 - Ex2\n");

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
			case 13:
				grey_fa();
				break;
			case 14:
				grey_fm();
				break;	
			case 15:
				create_colored_image();
				break;
			case 16:
				invert_matrix();
				break;
			case 21:
				canaleRGB();
				break;
			case 22:
				grayscale();
				break;
			case 23:
				binar();
				break;			
			case 24:
				convHSV();
				break;
//			case 31:
//				hist();
//				break;
//			case 32:
//				FDP();
//				break;
//			case 33:
//				showHistogram();
//				break;
			case 34:
				histNrRedAcc();
				break;
			case 35:
				praguriMultiple();
				break;
			case 36:
				applyFloydSteinbergErrorDiffusion();
				break;
			case 41:
				analyzeRegionProperties();
				break;
			case 42:
				analyzeAndDisplay();
				break;
		}
	}
	while (op!=0);
	return 0;
}