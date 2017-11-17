#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2\opencv.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp> 
#include <io.h>
#include <direct.h>
#include <limits>  




//#pragma comment(lib , "vl.lib")
using namespace std;
using namespace cv;


const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);


int entrmin = 100;
int entrmax = 1000;

int cannythresholdmin = 5;
int cannythresholdmax = 10;
int cannysize = 3;


void makedir(string &name)
{
	const char *tag;
	for (tag = name.c_str(); *tag; tag++)
	{
		if (*tag == '\\')
		{
			char buf[1000], path[1000];
			strcpy_s(buf, name.c_str());
			buf[strlen(name.c_str()) - strlen(tag) + 1] = NULL;
			strcpy_s(path, buf);
			if (_access(path, 6) == -1)
			{
				_mkdir(path);
			}
		}

	}
}


void openwritervideo(VideoWriter &writer, const string &name, int getfourcc, double srate, int frameW, int frameH)
{

	writer.open(name, getfourcc, srate, cvSize(frameW, frameH), true);

	if (!writer.isOpened())
	{
		cout << endl;
		cout << "111111111 open write video failed;" << endl;
		writer.open(name, -1, srate, cvSize(frameW, frameH), true);
	}


}

int getv1(string filename, string outputname1, string outputname2, int Gaussianparam = 7)
{
	makedir(outputname1);

	VideoCapture capture;
	bool readok = capture.open(filename);	//打开视频
	if (!readok)
	{
		cout << "read video failed!" << endl;
	}
	Mat frame, thresholdImage, output;

	//--------int history  为历史帧的数目 int nmixture 为高斯混合模型中SGM的数目 double backgroundRatio为背景比例 double noiseSigma 为噪声权重
	double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);	//获取帧数

	capture >> frame;
	if (frame.empty())
		cout << "video frame is empty" << endl;

	//double frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	//double frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	double frameW = frame.cols;
	double frameH = frame.rows;
	double getfourcc = capture.get(CV_CAP_PROP_FOURCC);

	//获取帧率
	double srate = capture.get(CV_CAP_PROP_FPS);

	//BackgroundSubtractorMOG2 bg_model;

	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();

	bgsubtractor->setHistory(50);
	bgsubtractor->setNMixtures(10);
	bgsubtractor->setBackgroundRatio(0.6);


	VideoWriter vwriter1;

	openwritervideo(vwriter1, outputname1, getfourcc, srate, frameW, frameH);

	VideoWriter vwriter2;

	openwritervideo(vwriter2, outputname2, getfourcc, srate, frameW, frameH);


	Mat mask1;
	int numtotal = 0;

	while (true)
	{
		capture >> frame;
		++numtotal;
		if (frame.empty())
			break;

		imshow("Origin image:", frame);

		Mat framemask;
		frame.copyTo(framemask);
		GaussianBlur(frame, frame, Size(Gaussianparam, Gaussianparam), 0, 0);
		//imshow("Gaussian", frame);

		//cout << " clos: " << frame.cols << "rows:  " << frame.rows << endl;
		bgsubtractor->apply(frame, mask1, 0.02);
		imshow("GMM_mask:", mask1);

		Mat qianjing;
		framemask.copyTo(qianjing, mask1);

		imshow("GMM_foregroung:：", qianjing);

		vwriter1 << qianjing;

		Mat graymat;
		cvtColor(qianjing, graymat, CV_BGR2GRAY);


		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		//imshow("graymat", graymat);
		if (waitKey(30) >= 0);
		threshold(graymat, graymat, 100, 255, THRESH_BINARY);
		if (waitKey(30) >= 0);
		//imshow("binary", graymat);
		graymat = graymat > 100;
		if (waitKey(30) >= 0);
		findContours(graymat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		//vwriter2 << qianjing;

		vector<Rect> orect;
		auto itContours = contours.begin();
		for (; itContours != contours.end();)
		{
			if (itContours->size() < entrmin)
				itContours = contours.erase(itContours);
			else
			{
				auto currect = boundingRect(Mat(*itContours));
				//rectangle(qianjing1, currect, Scalar(255, 0, 0), 2);
				orect.push_back(currect);
				++itContours;;
			}
		}

		for (int cj = 0; cj < mask1.rows; ++cj)
		{
			for (int ci = 0; ci < mask1.cols; ++ci)
			{
				if (0 != mask1.at<uchar>(cj, ci))
				{
					auto itcont = orect.begin();
					bool isin = false;
					while (itcont != orect.end())
					{
						//double curtest = pointPolygonTest(*itcont, Point2f(ci, cj), false);
						if ((*itcont).contains(Point2f(ci, cj)))
						{
							isin = true;
							break;
						}
						++itcont;
					}
					if (!isin)
					{
						mask1.at<uchar>(cj, ci) = 0;
					}
				}
			}
		}

		Mat qianjing1;
		framemask.copyTo(qianjing1, mask1);

		vwriter2 << qianjing1;

		imshow("clear foreground:", qianjing1);

		if (waitKey(20) == 27)
			break;
	}

	cout << "mask.channels : " << mask1.channels() << endl;

	Mat bgmat;
	bgsubtractor->getBackgroundImage(bgmat);

	imshow("bgm", bgmat);

	capture.release();
	vwriter1.release();
	vwriter2.release();

	return 0;
}


int getfg(string filename, string outputname1, string outputname2, string outputname3, string outputname4, string outputname5, int Gaussianparam = 7)
{



	char *fileName = ".\\1\\2\\3\\a.txt";
	const char *tag;
	for (tag = outputname1.c_str(); *tag; tag++)
	{
		if (*tag == '\\')
		{
			char buf[1000], path[1000];
			strcpy_s(buf, outputname1.c_str());
			buf[strlen(outputname1.c_str()) - strlen(tag) + 1] = NULL;
			strcpy_s(path, buf);
			if (_access(path, 6) == -1)
			{
				_mkdir(path);
			}
		}


	}

	VideoCapture capture;
	bool readok = capture.open(filename);	//打开视频
	if (!readok)
	{
		cout << "read video failed!" << endl;
	}
	Mat frame, thresholdImage, output;

	//--------int history  为历史帧的数目 int nmixture 为高斯混合模型中SGM的数目 double backgroundRatio为背景比例 double noiseSigma 为噪声权重
	double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);	//获取帧数

	capture >> frame;
	if (frame.empty())
		cout << "video frame is empty" << endl;

	//double frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	//double frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	double frameW = frame.cols;
	double frameH = frame.rows;
	double getfourcc = capture.get(CV_CAP_PROP_FOURCC);

	cout << "frame cols:  " << frameW << "  rows:  " << frameH << endl;

	//获取帧率
	double srate = capture.get(CV_CAP_PROP_FPS);

	//BackgroundSubtractorMOG2 bg_model;

	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();

	bgsubtractor->setHistory(50);
	bgsubtractor->setNMixtures(10);
	bgsubtractor->setBackgroundRatio(0.6);

	//Ptr<BackgroundSubtractorKNN> bgsubtractorknn = createBackgroundSubtractorKNN();
	//bgsubtractorknn->setkNNSamples(10);
	//bgsubtractorknn->setHistory(50);

	VideoWriter vwriter;
	//vwriter.open("./video/output1.avi", getfourcc, srate, cvSize(frameW, frameH), false);
	vwriter.open(outputname1, getfourcc, srate, cvSize(frameW, frameH), true);

	if (!vwriter.isOpened())
	{
		cout << endl;
		cout << "111111111 open write video failed;" << endl;
		vwriter.open(outputname1, -1, srate, cvSize(frameW, frameH), true);
	}

	VideoWriter vwriter2;
	//vwriter.open("./video/output1.avi", getfourcc, srate, cvSize(frameW, frameH), false);
	vwriter2.open(outputname2, getfourcc, srate, cvSize(frameW, frameH), true);

	if (!vwriter2.isOpened())
	{
		cout << endl;
		cout << "2222222 open write video failed;" << endl;
		vwriter2.open(outputname2, -1, srate, cvSize(frameW, frameH), true);
	}

	VideoWriter vwriter3;
	//vwriter.open("./video/output1.avi", getfourcc, srate, cvSize(frameW, frameH), false);
	vwriter3.open(outputname3, getfourcc, srate, cvSize(frameW, frameH), true);

	if (!vwriter3.isOpened())
	{
		cout << endl;
		cout << "3333333333 open write video failed;" << endl;
		vwriter3.open(outputname3, -1, srate, cvSize(frameW, frameH), true);
	}

	VideoWriter vwriter4;
	//vwriter.open("./video/output1.avi", getfourcc, srate, cvSize(frameW, frameH), false);
	vwriter4.open(outputname4, getfourcc, srate, cvSize(frameW, frameH), true);

	if (!vwriter4.isOpened())
	{
		cout << endl;
		cout << "44444444444 open write video failed;" << endl;
		vwriter4.open(outputname4, -1, srate, cvSize(frameW, frameH), true);
	}
	VideoWriter vwriter5;
	//vwriter.open("./video/output1.avi", getfourcc, srate, cvSize(frameW, frameH), false);
	vwriter5.open(outputname5, getfourcc, srate, cvSize(frameW, frameH), true);

	if (!vwriter5.isOpened())
	{
		cout << endl;
		cout << "555555 open write video failed;" << endl;
		vwriter5.open(outputname5, -1, srate, cvSize(frameW, frameH), true);
	}


	Mat mask1;

	int numtotal = 0;

	while (true)
	{
		capture >> frame;
		++numtotal;
		if (frame.empty())
			break;

		imshow("Origin image:", frame);

		Mat framemask;
		frame.copyTo(framemask);
		GaussianBlur(frame, frame, Size(Gaussianparam, Gaussianparam), 0, 0);
		//imshow("Gaussian", frame);



		//cout << " clos: " << frame.cols << "rows:  " << frame.rows << endl;
		bgsubtractor->apply(frame, mask1, 0.02);
		imshow("GMM_mask:", mask1);

		Mat qianjing;
		framemask.copyTo(qianjing, mask1);
		imshow("GMM_foregroung:：", qianjing);

		vwriter << qianjing;





		Mat qianjing1;
		qianjing.copyTo(qianjing1);

		//cout << "qianjing.channels:  " << qianjing.channels() << endl;

		Mat graymat;

		cvtColor(qianjing, graymat, CV_BGR2GRAY);

		//阈值较小的用于边缘连接 较大的值 用来控制强边缘的初始段
		//Canny(graymat, ermat1, cannythresholdmin, cannythresholdmax, cannysize);
		//cout << "graymat.channels:  " << graymat.channels() << endl;
		//imshow("graymat", graymat);
		//blur(graymat, graymat, Size(3, 3));
		//对前景进行提取  去除掉比较小的轮廓；

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		//Canny(graymat, graymat, cannythresholdmin, cannythresholdmax, cannysize);
		//findContours(mask1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		//imshow("graymat", graymat);
		if (waitKey(30) >= 0);
		threshold(graymat, graymat, 100, 255, THRESH_BINARY);
		if (waitKey(30) >= 0);
		//imshow("binary", graymat);
		graymat = graymat > 100;
		if (waitKey(30) >= 0);
		findContours(graymat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		//cout << " number of entr external:  " << contours.size() << endl;

		drawContours(qianjing, contours, -1, Scalar(255, 255, 255), 2);


		imshow("提取轮廓：", qianjing);

		vwriter2 << qianjing;

		vector<Rect> orect;



		auto itContours = contours.begin();
		for (; itContours != contours.end();)
		{
			if (itContours->size() < entrmin)
				itContours = contours.erase(itContours);
			else
			{
				auto currect = boundingRect(Mat(*itContours));
				//rectangle(qianjing1, currect, Scalar(255,0,0), 2);
				orect.push_back(currect);
				++itContours;;
			}
		}


		Mat grabcutmask;
		mask1.copyTo(grabcutmask);
		// 使用grabcut时，mask必须为以下四者之一
		//************************** GCD_BGD(=0) background 
		////************************** GCD_FGD(=1) foreground
		////************************** GCD_PR_BGD(=2） probably background 
		////************************** GCD_PR_FGD(=3) probably foreground
		int bgnum = 0;
		int fgnum = 0;
		for (int cj = 0; cj < grabcutmask.rows; ++cj)
		{
			for (int ci = 0; ci < grabcutmask.cols; ++ci)

			if (200 < grabcutmask.at<uchar>(cj, ci))
			{
				auto itcont = contours.begin();
				bool isin = false;
				while (itcont != contours.end())
				{
					double curtest = pointPolygonTest(*itcont, Point2f(ci, cj), false);
					if (curtest >= 0)
					{
						++fgnum;
						isin = true;
						break;
					}
					++itcont;
				}
				if (isin)
				{
					grabcutmask.at<uchar>(cj, ci) = 1;
				}
				else
				{
					grabcutmask.at<uchar>(cj, ci) = 3;
				}

			}
			else
			{
				if (100 > grabcutmask.at<uchar>(cj, ci))
				{
					grabcutmask.at<uchar>(cj, ci) = 0;
					++bgnum;
				}
				else
				{
					grabcutmask.at<uchar>(cj, ci) = 2;
				}
			}
		}

		Mat bgdModel, fgdModel;
		if (0 < fgnum && 0 < bgnum)
		{
			grabCut(frame, grabcutmask, Rect(), bgdModel, fgdModel, 3, GC_INIT_WITH_MASK);
		}
		else
		{
			grabCut(frame, grabcutmask, Rect(), bgdModel, fgdModel, 3, GC_EVAL);
		}
		auto curp = grabcutmask.begin<uchar>();
		while (curp != grabcutmask.end<uchar>())
		{
			if (0 == (*curp))
			{
				*curp = 0;
			}
			if (1 == (*curp))
			{
				*curp = 255;
			}
			if (2 == (*curp))
			{
				*curp = 0;
			}
			if (3 == (*curp))
			{
				*curp = 127;
			}
			++curp;
		}
		Mat grabfr;
		frame.copyTo(grabfr, grabcutmask);

		vwriter4 << grabfr;

		imshow("grabcut foreground", grabfr);

		imshow("包围矩形：", qianjing1);

		drawContours(qianjing1, contours, -1, Scalar(255, 255, 255), 2);

		imshow("裁剪轮廓：", qianjing1);
		vwriter3 << qianjing1;
		for (int cj = 0; cj < mask1.rows; ++cj)
		{
			for (int ci = 0; ci < mask1.cols; ++ci)
			{

				if (0 != mask1.at<uchar>(cj, ci))
				{
					auto itcont = contours.begin();
					bool isin = false;
					while (itcont != contours.end())
					{
						double curtest = pointPolygonTest(*itcont, Point2f(ci, cj), false);
						if (curtest >= 0)
						{
							isin = true;
							break;
						}
						++itcont;
					}
					if (!isin)
					{
						mask1.at<uchar>(cj, ci) = 0;
					}


				}
			}
		}


		framemask.copyTo(qianjing1, mask1);
		vwriter5 << qianjing1;

		imshow("clear foreground:", qianjing1);
		framemask.setTo(0, mask1);

		if (waitKey(20) == 27)
			break;
	}

	cout << "mask.channels : " << mask1.channels() << endl;

	Mat bgmat;
	bgsubtractor->getBackgroundImage(bgmat);

	imshow("bgm", bgmat);
	//bgmat.convertTo(bgmat,CV_8UC3);
	//imwrite("1.jpg", bgmat);
	//Mat bgmat1;
	//bgsubtractorknn->getBackgroundImage(bgmat1);
	////imwrite(bgname2, bgmat1);
	//imshow("bgm1", bgmat1);

	capture.release();
	vwriter.release();
	vwriter2.release();
	vwriter3.release();
	vwriter4.release();
	vwriter5.release();
	//vwriter2.release();

	return 0;
}

void getcapture()
{


	VideoCapture  capture1;

	capture1.open(0);
	while (capture1.isOpened())
	{
		Mat mat1;
		capture1 >> mat1;
		int i = 0;
		if (!mat1.empty())
		{

			char *name1 = new char[40];
			sprintf_s(name1, 40, "video1%i.png", i);
			imshow(name1, mat1);
			imwrite(name1, mat1);
			i++;
			if (waitKey(30) >= 0) break;
		}

	}

}


#define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))


int main()
{

	int total = 4;
	string s = "jiankong";

	for (int i = 1; i < total; ++i)
	{
		char *filename = new char[40];
		sprintf_s(filename, 40, ".\\video\\%d.avi",  i);
		cout << filename << endl;
		char *outputname1 = new char[40];
		sprintf_s(outputname1, 40, ".\\video\\output\\videoGMM%d.avi",  i);
		char *outputname2 = new char[40];
		sprintf_s(outputname2, 40, ".\\video\\output\\videocotor%d.avi",  i);
		char *outputname3 = new char[50];
		sprintf_s(outputname3, 50, ".\\video\\output\\videocotorthrreshold%d.avi",  i);
		char *outputname4 = new char[50];
		sprintf_s(outputname4, 50, ".\\video\\output\\videograbcut%d.avi",  i);
		char *outputname5 = new char[50];
		sprintf_s(outputname5, 50, ".\\video\\output\\videoclear%d.avi", i);
		getfg(filename, outputname1, outputname2, outputname3, outputname4, outputname5);

		//char *GMM_get_name = new char[50];
		//sprintf_s(GMM_get_name, 50, ".\\video\\%s\\output\\GMM_get_%d.avi", s.c_str(), i);
		//char *GMM_clear_get_name = new char[50];
		//sprintf_s(GMM_clear_get_name, 50, ".\\video\\%s\\output\\GMM_clear_get_name_%d.avi", s.c_str(), i);
		//getv1(filename, GMM_get_name, GMM_clear_get_name);

	}


	system("pause");
}




//void Vlfeat_GMM()
//{
//
//	//   double test_data[4][3] = {
//	//       49, 49, 49,
//	//       34, 24, 16,
//	//       5.0, 6.2, 8.4,
//	//       10.3, 10.4, 10.5
//	//   };
//	//VideoCapture capture;
//	//VideoWriter vwriter;
//	//std::cout << "Load raw data from '../indata/'..." << std::flush;
//	//cout << endl;
//	//bool readok = capture.open("./video/1.avi");	//打开视频
//	//if (!readok)
//	//{
//	//	cout << "open video failed!" << endl;
//	//}
//	//int totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);	//获取帧数
//	//int frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
//	//int frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);
//	////获取帧率
//	//double srate = capture.get(CV_CAP_PROP_FPS);
//	////两帧间的间隔时间:
//	//double delay = 1000 / srate;
//	//Mat currentmat;
//	//capture >> currentmat;
//	//int rownum = currentmat.rows;
//	//int colnum = currentmat.cols;
//	//int channelsnum = currentmat.channels();
//	//const int dim = 3;   //Dimension of feature
//	//const int cluster_num = 12; //Cluster number
//	//const int size = rownum*colnum; //Number of samples
//	//cout << "start assignment" << endl;
//	//double *dataset = new double[rownum*colnum*channelsnum];
//	//for (int i = 0; i < rownum; i++)
//	//{
//	//	uchar *data = currentmat.ptr<uchar>(i);
//	//	
//	//	for (int j = 0; j < colnum; j++)
//	//	{
//	//		dataset[3*(i*j + j)+0] = data[3 * j];
//	//		//cout << (int)data[3 * j] << endl;
//	//		//cout << (int)dataset(i*j + j, 0) << endl;
//	//		dataset[3*(i*j + j)+1] = data[3 * j + 1];
//	//		dataset[3*(i*j + j)+2] = data[3 * j + 2];
//	//	}
//	//}
//	//cout << "end assignment" << endl;
//	/////////////////////  end get data
//	////Test GMM
//	//cout << "GMM starty " << endl;
//	//GMM *gmm = new GMM(dim, cluster_num); //GMM has 3 SGM
//	//cout << "GMM end " << endl;
//	//cout << "GMM start train" << endl;
//	//   gmm->Train(dataset,size); //Training GMM
//	//cout << "GMM finish train" << endl;
//	//   printf("\nTest GMM:\n");
//	//   for(int i = 0; i < 4; ++i)
//	//{
//	//    printf("The Probability of %f, %f, %f  is %f \n",test_data[i][0],test_data[i][1],test_data[i][2],gmm->GetProbability(test_data[i]));
//	//}
//	//   //save GMM to file
//	//ofstream gmm_file("gmm.txt");
//	//assert(gmm_file);
//	//gmm_file<<*gmm;
//	//gmm_file.close();
//
//
//	VideoCapture capture;
//	VideoWriter vwriter;
//	std::cout << "Load raw data from '../indata/'..." << std::flush;
//	cout << endl;
//	bool readok = capture.open("./video/1.avi");	//打开视频
//	if (!readok)
//	{
//		cout << "open video failed!" << endl;
//	}
//	double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);	//获取帧数
//	double frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
//	double frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);
//	//获取帧率
//	double srate = capture.get(CV_CAP_PROP_FPS);
//	//两帧间的间隔时间:
//	double delay = 1000 / srate;
//	Mat currentmat;
//	capture >> currentmat;
//	int rownum = currentmat.rows;
//	int colnum = currentmat.cols;
//	int channelsnum = currentmat.channels();
//	const int dim = 3;   //Dimension of feature
//	const int cluster_num = 12; //Cluster number
//	const int size = rownum*colnum; //Number of samples
//	cout << "start assignment" << endl;
//	double *dataset = new double[rownum*colnum*channelsnum];
//	for (int i = 0; i < rownum; i++)
//	{
//		uchar *data = currentmat.ptr<uchar>(i);
//
//		for (int j = 0; j < colnum; j++)
//		{
//			dataset[3 * (i*j + j) + 0] = data[3 * j];
//			//cout << (int)data[3 * j] << endl;
//			//cout << (int)dataset(i*j + j, 0) << endl;
//			dataset[3 * (i*j + j) + 1] = data[3 * j + 1];
//			dataset[3 * (i*j + j) + 2] = data[3 * j + 2];
//		}
//	}
//	cout << "end assignment" << endl;
//	VlGMM *gmm = vl_gmm_new(VL_TYPE_DOUBLE, dim, cluster_num);
//	vl_gmm_set_max_num_iterations(gmm, 100);
//	vl_gmm_set_initialization(gmm, VlGMMRand);
//	vl_gmm_cluster(gmm, dataset, size);
//
//	float * means;//respectively the means  
//	float * covariances;//diagonal covariance matrices Σk
//	float * priors;// and prior probabilities πk of the numClusters Gaussian modes.
//	float * posteriors;
//	double loglikelihood;
//
//	// get the means, covariances, and priors of the GMM
//	means = (float *)vl_gmm_get_means(gmm);
//	for (int i = 0; i < cluster_num; i++)
//	{
//		cout << "means" << i << ":  " << means[i] << endl;
//	}
//	covariances = (float *)vl_gmm_get_covariances(gmm);
//	priors = (float *)vl_gmm_get_priors(gmm);
//	// get loglikelihood of the estimated GMM
//	loglikelihood = vl_gmm_get_loglikelihood(gmm);
//	// get the soft assignments of the data points to each cluster
//	posteriors = (float *)vl_gmm_get_posteriors(gmm);
//
//	VL_PRINT("Hello world!\n");
//
//
//}