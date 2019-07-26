////
// Compare with smiledetect.cpp sample provided with opencv
////
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

// TBB NOTE: we need these headers
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
volatile bool done = false; // volatile is enough here. We don't need a mutex for this simple flag.

struct ProcessingChainData
{
	cv::Mat img;
	vector<cv::Rect> faces, faces2;
	cv::Mat gray, smallImg;
};

struct time_step
{
	string filter;
	string execution_time;
};
void detectAndDrawTBB(cv::VideoCapture &capture,
	tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue,
	cv::CascadeClassifier& cascade,
	cv::CascadeClassifier& nestedCascade,
	double scale, bool tryflip, vector<time_step>& time_process);

string cascadeName;
string nestedCascadeName;

int main()
{
	cv::VideoCapture capture;
	cv::Mat frame, image;
	string inputName;
	bool tryflip;

	// TBB NOTE: these are not thread safe, so be careful not to use them in parallel.
	cv::CascadeClassifier cascade, nestedCascade;
	double scale;

	cascadeName = "haarcascade_frontalface_alt.xml";
	nestedCascadeName = "haarcascade_smile.xml";
	cascade.load(cascadeName);
	nestedCascade.load(nestedCascadeName);
	tryflip = false;
	inputName = "short_test.mp4";
	scale = 2;
	capture.open(inputName);
	vector<time_step> time_process;
	int64 startTime;
	if (capture.isOpened())
	{
		cout << "Video capturing has been started ..." << endl;
		cout << endl << "NOTE: Smile intensity will only be valid after a first smile has been detected" << endl;

		tbb::concurrent_bounded_queue<ProcessingChainData *> guiQueue;
		guiQueue.set_capacity(2); // TBB NOTE: flow control so the pipeline won't fill too much RAM
		std::thread pipelineRunner = thread(detectAndDrawTBB, ref(capture), ref(guiQueue), ref(cascade), ref(nestedCascade), scale, tryflip, ref(time_process));

		startTime = cv::getTickCount();

		// TBB NOTE: GUI is executed in main thread
		ProcessingChainData *pData = 0;
		for (; !done;)
		{
			if (guiQueue.try_pop(pData))
			{
				char c = (char)cv::waitKey(1);
				if (c == 27 || c == 'q' || c == 'Q')
				{
					done = true;
				}
				imshow("result", pData->img);
				delete pData;
				pData = 0;
			}
		}
		double tfreq = cv::getTickFrequency();
		double secs = ((double)cv::getTickCount() - startTime) / tfreq;
		for (vector<time_step>::iterator it = time_process.begin(); it < time_process.end(); it++)
		{
			cout << "Filter: " << it->filter << " executed in :" << it->execution_time << endl;
		}
		cout << "Execution took " << fixed << secs << " seconds." << endl;
		// TBB NOTE: flush the queue after marking 'done'
		do
		{
			delete pData;
		} while (guiQueue.try_pop(pData));
		pipelineRunner.join(); // TBB NOTE: wait for the pipeline to finish
	}
	else
	{
		cerr << "ERROR: Could not initiate capture" << endl;
		return -1;
	}

	return 0;
}

void detectAndDrawTBB(cv::VideoCapture &capture,
	tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue,
	cv::CascadeClassifier& cascade,
	cv::CascadeClassifier& nestedCascade,
	double scale, bool tryflip, vector<time_step>& time_process)
{
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
	
	int number_filter = 7;
	tbb::parallel_pipeline(number_filter, // TBB NOTE: (recommendation) NumberOfFilters
							  // 1st filter
		tbb::make_filter<void, ProcessingChainData*>(tbb::filter::serial_in_order,
			[&](tbb::flow_control& fc)->ProcessingChainData*
	{   // TBB NOTE: this filter feeds input into the pipeline
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		cv::Mat frame;
		capture >> frame;
		if (done || frame.empty())
		{
			// 'done' is our own exit flag
			// being set and checked in and out
			// of the pipeline
			done = true;

			// These 2 lines are how to tell TBB to stop the pipeline
			fc.stop();
			return 0;
		}
		auto pData = new ProcessingChainData;
		pData->img = frame.clone();
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "1st";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
		return pData;
	}
			)&
		// 2nd filter
		tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
			[&](ProcessingChainData *pData)->ProcessingChainData*
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		cvtColor(pData->img, pData->gray, cv::COLOR_BGR2GRAY);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "2nd";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
		return pData;
	}
			)&
		// 3rd filter
		tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
			[&](ProcessingChainData *pData)->ProcessingChainData*
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		double fx = 1 / scale;
		resize(pData->gray, pData->smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "3rd";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
		return pData;
	}
			)&
		// 4th filter
		tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
			[&](ProcessingChainData *pData)->ProcessingChainData*
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		equalizeHist(pData->smallImg, pData->smallImg);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "4th";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
		return pData;
	}
			)&
		// 5th filter
		tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
			[&](ProcessingChainData *pData)->ProcessingChainData*
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		cascade.detectMultiScale(pData->smallImg, pData->faces,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| cv::CASCADE_SCALE_IMAGE,
			cv::Size(30, 30));
		if (tryflip)
		{   // TBB NOTE: 1. CascadeClassifier is already paralleled by OpenCV
			//           2. Is is not thread safe, so don't call the same classifier from different threads.
			flip(pData->smallImg, pData->smallImg, 1);
			cascade.detectMultiScale(pData->smallImg, pData->faces2,
				1.1, 2, 0
				//|CASCADE_FIND_BIGGEST_OBJECT
				//|CASCADE_DO_ROUGH_SEARCH
				| cv::CASCADE_SCALE_IMAGE,
				cv::Size(30, 30));
			for (vector<cv::Rect>::const_iterator r = pData->faces2.begin(); r != pData->faces2.end(); ++r)
			{
				pData->faces.push_back(cv::Rect(pData->smallImg.cols - r->x - r->width, r->y, r->width, r->height));
			}
		}
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "5th";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
		return pData;
	}
			)&
		// 6th filter
		tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
			[&](ProcessingChainData *pData)->ProcessingChainData*
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		for (size_t i = 0; i < pData->faces.size(); i++)
		{
			cv::Rect r = pData->faces[i];
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
				circle(pData->img, center, radius, color, 3, 8, 0);
			}
			else
				rectangle(pData->img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
					cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
					color, 3, 8, 0);

			const int half_height = cvRound((float)r.height / 2);
			r.y = r.y + half_height;
			r.height = half_height - 1;
			smallImgROI = pData->smallImg(r);
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

			// Draw rectangle on the left side of the image reflecting smile intensity
			float intensityZeroOne = ((float)smile_neighbors - min_neighbors) / (max_neighbors - min_neighbors + 1);
			int rect_height = cvRound((float)pData->img.rows * intensityZeroOne);
			cv::Scalar col = cv::Scalar((float)255 * intensityZeroOne, 0, 0);
			rectangle(pData->img, cvPoint(0, pData->img.rows), cvPoint(pData->img.cols / 10, pData->img.rows - rect_height), col, -1);
		}
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "6th";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
		return pData;
	}
			)&
		// 7th filter
		tbb::make_filter<ProcessingChainData*, void>(tbb::filter::serial_in_order,
			[&](ProcessingChainData *pData)
	{   // TBB NOTE: pipeline end point. dispatch to GUI
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		if (!done)
		{
			try
			{
				guiQueue.push(pData);
			}
			catch (...)
			{
				cout << "Pipeline caught an exception on the queue" << endl;
				done = true;
			}
		}
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		string duration_result = to_string(duration);
		time_step ts;
		ts.filter = "7th";
		ts.execution_time = duration_result;
		time_process.push_back(ts);
	}
			)
		);
}
