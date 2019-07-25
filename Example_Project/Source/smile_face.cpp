#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;

void detectAndDraw(cv::Mat& img, cv::CascadeClassifier& cascade,
	cv::CascadeClassifier& nestedCascade,
	double scale, bool tryflip);

string cascadeName;
string nestedCascadeName;

int main()
{
	cv::VideoCapture capture;
	cv::Mat frame, image;
	string inputName;
	bool tryflip;


	cv::CascadeClassifier cascade, nestedCascade;
	double scale;

	cascadeName = "haarcascade_frontalface_alt.xml";
	nestedCascadeName = "haarcascade_smile.xml";
	cascade.load(cascadeName);
	nestedCascade.load(nestedCascadeName);
	tryflip = false;
	inputName = "test.mp4";
	scale = 2;
	capture.open(inputName);
	int64 startTime;
	if (capture.isOpened())
	{
		cout << "Video capturing has been started ..." << endl;
		cout << endl << "NOTE: Smile intensity will only be valid after a first smile has been detected" << endl;

		startTime = cv::getTickCount();

		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;

			cv::Mat frame1 = frame.clone();
			detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip);

			char c = (char)cv::waitKey(1);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	double tfreq = cv::getTickFrequency();
	double secs = ((double)cv::getTickCount() - startTime) / tfreq;
	cout << "Execution took " << fixed << secs << " seconds." << endl;
	return 0;
}

void detectAndDraw(cv::Mat& img, cv::CascadeClassifier& cascade,
	cv::CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	vector<cv::Rect> faces, faces2;
	const static cv::Scalar colors[] =
	{
		cv::Scalar(255,0,0),
		cv::Scalar(255,128,0),
		cv::Scalar(255,255,0),
		cv::Scalar(0,255,0),
		cv::Scalar(0,128,255),
		cv::Scalar(0,255,255),
		cv::Scalar(0,0,255),
		cv::Scalar(255,0,255)
	};
	cv::Mat gray, smallImg;

	cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	double fx = 1 / scale;
	resize(gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| cv::CASCADE_SCALE_IMAGE,
		cv::Size(30, 30));
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| cv::CASCADE_SCALE_IMAGE,
			cv::Size(30, 30));
		for (vector<cv::Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(cv::Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}

	for (size_t i = 0; i < faces.size(); i++)
	{
		cv::Rect r = faces[i];
		cv::Mat smallImgROI;
		vector<cv::Rect> nestedObjects;
		cv::Point center;
		cv::Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			cv::rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
				cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);

		const int half_height = cvRound((float)r.height / 2);
		r.y = r.y + half_height;
		r.height = half_height - 1;
		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 0, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| cv::CASCADE_SCALE_IMAGE,
			cv::Size(30, 30));

		// The number of detected neighbors depends on image size (and also illumination, etc.). The
		// following steps use a floating minimum and maximum of neighbors. Intensity thus estimated will be
		//accurate only after a first smile has been displayed by the user.
		const int smile_neighbors = (int)nestedObjects.size();
		static int max_neighbors = -1;
		static int min_neighbors = -1;
		if (min_neighbors == -1) min_neighbors = smile_neighbors;
		max_neighbors = MAX(max_neighbors, smile_neighbors);

		// Draw cv::Rectangle on the left side of the image reflecting smile intensity
		float intensityZeroOne = ((float)smile_neighbors - min_neighbors) / (max_neighbors - min_neighbors + 1);
		int Rect_height = cvRound((float)img.rows * intensityZeroOne);
		cv::Scalar col = cv::Scalar((float)255 * intensityZeroOne, 0, 0);
		cv::rectangle(img, cvPoint(0, img.rows), cvPoint(img.cols / 10, img.rows - Rect_height), col, -1);
	}

	imshow("result", img);
	cv::waitKey(1);
}
