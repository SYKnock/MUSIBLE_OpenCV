#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <math.h>

using namespace cv;
using namespace std;

typedef struct __Braille
{
	Rect rect;
	int value;
	int index;

	__Braille()
	{
		value = 0;
		index = 0;
	}
	~__Braille()
	{
	}

} Braille;

typedef struct __Note
{
	int tick;
	int value;
	__Note()
	{
		tick = 0;
		value = 0;
	}
	~__Note()
	{
	}

} Note;

vector<Braille> brailleSet;
vector<Note> noteSet;
int xLineCnt = 1;
int yLineCnt = 1;
int realX;

Mat preprocess(Mat score);
Mat brailleSegmentation(Mat score, vector<int> blobX, vector<int> blobY, int blobSize);
Mat reblobWithSegmentation(Mat blobScore, vector<KeyPoint> keypoints);
Mat findCircle(Mat score);
vector<int> makeGridX(vector<int> blobX, int blobSize);
Mat dataCheck(Mat score, vector<KeyPoint> keypoints);
float getBlobSize(vector<KeyPoint> keypoints);
void convert2Score();
int checkOctave(int octave);

Mat preprocess(Mat score)
{
	Mat aThresholdScore, erodeScore, gaussianScore, thresholdScore, dilateScore;

	// change to binary
	adaptiveThreshold(score, aThresholdScore, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, 10);

	// gaussianblur
	GaussianBlur(aThresholdScore, gaussianScore, Size(3, 3), 0);

	// opening
	erode(gaussianScore, erodeScore, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1));
	dilate(erodeScore, dilateScore, Mat::ones(Size(1, 1), CV_8UC1), Point(-1, -1));

	// final threshold
	threshold(dilateScore, thresholdScore, 71, 255, THRESH_BINARY);

	return thresholdScore;
}

Mat brailleSegmentation(Mat score, vector<int> gridX, vector<int> gridY, int blobSize)
{
	Mat BrailleScore = Mat(score.size(), CV_8UC3);
	cvtColor(score, BrailleScore, COLOR_GRAY2BGR);

	Braille tempB;
	int value = 0;
	int index = 0;

	for (int i = 0; i < gridY.size() - 2; i += 3)
	{
		for (int j = 0; j < gridX.size() - 1; j += 2)
		{
			Rect rect = Rect(Point(gridX[j] - (blobSize * 1 / 2), gridY[i] - (blobSize * 1 / 2)), Point(gridX[j + 1] + (blobSize * 1 / 2), gridY[i + 2] + (blobSize * 1 / 2)));
			rectangle(BrailleScore, rect, Scalar(0, 0, 255));
			value = 0;
			for (int y = 0; y < 3; y++)
			{
				for (int x = 0; x < 2; x++)
				{
					if (score.at<uchar>(Point((int)gridX[j + x], (int)gridY[i + y])) == 0)
						value++;
					value = value << 1;
				}
			}
			value = value >> 1;
			tempB.rect = rect;
			tempB.index = index++;
			tempB.value = value;
			brailleSet.push_back(tempB);
		}
	}

	// namedWindow("Result5", 0);
	// imshow("Result5", BrailleScore);
	// resizeWindow("Result5", BrailleScore.cols, BrailleScore.rows);

	return BrailleScore;
}

float getBlobSize(vector<KeyPoint> keypoints)
{
	float blobSize = 0.0;
	for (int i = 0; i < keypoints.size(); i++)
		blobSize += keypoints[i].size;

	blobSize = blobSize / keypoints.size();
	return blobSize;
}

Mat reblobWithSegmentation(Mat blobScore, vector<KeyPoint> keypoints)
{
	float blobSize = getBlobSize(keypoints);

	vector<int> blobX;
	vector<int> blobY;

	bool flag = true;
	int representLineSize = 0;

	vector<int> tempX;
	vector<int> tempY;

	for (int i = 0; i < keypoints.size(); i++)
	{
		tempX.push_back((int)keypoints[i].pt.x);
		tempY.push_back((int)keypoints[i].pt.y);
	}
	sort(tempX.begin(), tempX.end());
	sort(tempY.begin(), tempY.end());

	for (int i = 0; i < keypoints.size(); i++)
	{
		if (i != 0)
		{
			if (tempX[i - 1] + 4 < tempX[i])
				xLineCnt++;
			if (tempY[i - 1] + 4 < tempY[i])
				yLineCnt++;
		}
	}

	int *avgX = new int[xLineCnt];
	int *avgY = new int[yLineCnt];
	memset(avgX, 0, xLineCnt * sizeof(float));
	memset(avgY, 0, yLineCnt * sizeof(float));
	int *tmpCntX = new int[xLineCnt];
	int *tmpCntY = new int[yLineCnt];
	memset(tmpCntX, 0, xLineCnt * sizeof(int));
	memset(tmpCntY, 0, yLineCnt * sizeof(int));
	int tmpIndexX = 0;
	int tmpIndexY = 0;

	for (int i = 0; i < keypoints.size(); i++)
	{
		if (i == 0)
		{
			avgX[0] += tempX[0];
			avgY[0] += tempY[0];
			tmpCntX[0]++;
			tmpCntY[0]++;
		}
		else
		{
			if (abs(tempX[i] - tempX[i - 1]) < 4)
			{
				avgX[tmpIndexX] += tempX[i];
				tmpCntX[tmpIndexX]++;
			}
			else
			{
				tmpIndexX++;
				avgX[tmpIndexX] += tempX[i];
				tmpCntX[tmpIndexX]++;
			}

			if (abs(tempY[i] - tempY[i - 1]) < 4)
			{
				avgY[tmpIndexY] += tempY[i];
				tmpCntY[tmpIndexY]++;
			}
			else
			{
				tmpIndexY++;
				avgY[tmpIndexY] += tempY[i];
				tmpCntY[tmpIndexY]++;
			}
		}
	}
	for (int i = 0; i < xLineCnt; i++)
		avgX[i] /= tmpCntX[i];

	for (int i = 0; i < yLineCnt; i++)
		avgY[i] /= tmpCntY[i];

	for (int i = 0; i < xLineCnt; i++)
		blobX.push_back((int)avgX[i]);
	for (int i = 0; i < yLineCnt; i++)
		blobY.push_back((int)avgY[i]);

	sort(blobX.begin(), blobX.end());
	sort(blobY.begin(), blobY.end());

	Mat coordinateScore = blobScore.clone();

	for (int i = 0; i < blobX.size(); i++)
		line(coordinateScore, Point(blobX[i], 0), Point(blobX[i], coordinateScore.rows), Scalar(255, 0, 0));
	for (int i = 0; i < blobY.size(); i++)
		line(coordinateScore, Point(0, blobY[i]), Point(coordinateScore.cols, blobY[i]), Scalar(255, 0, 0));

	// namedWindow("Result3", 0);
	// imshow("Result3", coordinateScore);
	// resizeWindow("Result3", coordinateScore.cols, coordinateScore.rows);

	int distanceX;
	int distanceY;
	int xBuffer;
	int yBuffer;

	for (int i = 0; i < keypoints.size(); i++)
	{
		distanceX = blobScore.cols / 2;
		distanceY = blobScore.rows / 2;
		xBuffer = 0;
		yBuffer = 0;

		for (int j = 0; j < blobX.size(); j++)
		{
			if (distanceX > abs(keypoints[i].pt.x - blobX[j]))
			{
				distanceX = abs(keypoints[i].pt.x - blobX[j]);
				xBuffer = blobX[j];
			}
		}
		keypoints[i].pt.x = xBuffer;

		for (int j = 0; j < blobY.size(); j++)
		{
			if (distanceY > abs(keypoints[i].pt.y - blobY[j]))
			{
				distanceY = abs(keypoints[i].pt.y - blobY[j]);
				yBuffer = blobY[j];
			}
		}
		keypoints[i].pt.y = yBuffer;
	}

	Mat fixedScore = Mat(blobScore.size(), CV_8UC1);
	fixedScore.setTo(255);
	for (int i = 0; i < keypoints.size(); i++)
	{
		circle(fixedScore, Point(keypoints[i].pt.x, keypoints[i].pt.y), blobSize / 2, Scalar(0), -1, LINE_AA);
	}

	// namedWindow("Result4", 0);
	// imshow("Result4", fixedScore);
	// resizeWindow("Result4", fixedScore.cols, fixedScore.rows);

	vector<int> gridX = makeGridX(blobX, blobSize);
	vector<int> gridY;

	for (vector<int>::iterator iter = blobY.begin(); iter != blobY.end(); iter++)
	{
		gridY.push_back(*iter);
	}

	Mat tmp = fixedScore.clone();
	cvtColor(tmp, tmp, COLOR_GRAY2BGR);

	for (int i = 0; i < gridX.size(); i++)
		line(tmp, Point(gridX[i], 0), Point(gridX[i], tmp.rows), Scalar(255, 0, 0));
	for (int i = 0; i < gridY.size(); i++)
		line(tmp, Point(0, gridY[i]), Point(tmp.cols, gridY[i]), Scalar(255, 0, 0));

	// namedWindow("Result6", 0);
	// imshow("Result6", tmp);
	// resizeWindow("Result6", tmp.cols, tmp.rows);

	Mat BrailleScore = brailleSegmentation(fixedScore, gridX, gridY, blobSize);

	return BrailleScore;
}

vector<int> makeGridX(vector<int> blobX, int blobSize)
{
	vector<int> gridX;
	int average = 0;
	int *distance = new int[xLineCnt - 1];

	for (vector<int>::iterator iter = blobX.begin(); iter != blobX.end(); iter++)
		gridX.push_back(*iter);

	for (int i = xLineCnt - 2; i >= 0; i--)
		distance[i] = blobX[i + 1] - blobX[i];
	for (int i = 0; i < xLineCnt - 1; i++)
		average += distance[i];
	average /= xLineCnt - yLineCnt;

	vector<int> refinedDistance;

	for (int i = 0; i < xLineCnt - 1; i++)
	{
		if (distance[i] > average + blobSize)
			continue;
		else
		{
			refinedDistance.push_back(distance[i]);
		}
	}
	int refinedAverage = 0;

	for (vector<int>::iterator iter = refinedDistance.begin(); iter != refinedDistance.end(); iter++)
		refinedAverage += *iter;

	refinedAverage /= refinedDistance.size();

	for (int i = 0; i < xLineCnt - 1; i++)
	{
		if (gridX[i + 1] - gridX[i] >= refinedAverage + blobSize)
		{
			gridX.push_back(gridX[i] + refinedAverage);
			sort(gridX.begin(), gridX.end());
			i--;
		}
	}
	realX = gridX.size();
	return gridX;
}

Mat findCircle(Mat score)
{
	Mat blobScore(score.size(), CV_8UC3);
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 20;
	params.maxThreshold = 200;

	// filter by black dots
	params.filterByColor = true;
	params.blobColor = 0;

	// filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.5;

	// filter by dot size
	params.filterByArea = true;
	params.minArea = 4;
	params.maxArea = 1200;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> keypoints;
	detector->detect(score, keypoints);

	if (keypoints.empty())
	{
		cout << "There is no braille in score" << endl;
		return score;
	}

	drawKeypoints(score, keypoints, blobScore, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// namedWindow("Result2", 0);
	// imshow("Result2", blobScore);
	// resizeWindow("Result2", blobScore.cols, blobScore.rows);

	// blob is done

	// now make new dot image & do the segmentation
	Mat BrailleScore = reblobWithSegmentation(blobScore, keypoints);

	Mat DataScore(score.size(), CV_8UC3);
	DataScore = dataCheck(BrailleScore, keypoints);

	return DataScore;
}

Mat dataCheck(Mat score, vector<KeyPoint> keypoints)
{
	float blobSize = getBlobSize(keypoints);

	Mat dataScore = Mat(score.size(), CV_8UC3);
	dataScore.setTo(255);
	addWeighted(dataScore, 0.8, score, 0.2, 0.0, dataScore);

	int font = FONT_HERSHEY_SIMPLEX;
	double fontScale = brailleSet[0].rect.size().width / 60.0;
	int fontThick = (int)std::round(fontScale * 2);

	for (int i = 0; i < brailleSet.size(); i++)
	{
		Point center, bottomLeft;
		center = (brailleSet[i].rect.tl() + brailleSet[i].rect.br()) / 2;
		center.x -= getTextSize(to_string(brailleSet[i].value), font, fontScale, fontThick, 0).width / 2;
		center.y += getTextSize(to_string(brailleSet[i].value), font, fontScale, fontThick, 0).height / 2;

		bottomLeft = Point(brailleSet[i].rect.tl().x, brailleSet[i].rect.br().y);
		bottomLeft.x -= blobSize / 2;
		bottomLeft.y += getTextSize(bitset<6>(brailleSet[i].value).to_string(), font, fontScale * 0.7, fontThick * 0.7, 0).height / 2 + blobSize / 2;

		putText(dataScore, to_string(brailleSet[i].value), center, font, fontScale, Scalar(255, 0, 0), fontThick);
		putText(dataScore, bitset<6>(brailleSet[i].value).to_string(), bottomLeft, font, fontScale * 0.5, Scalar(0, 0, 0), fontThick * 0.5);
	}

	// namedWindow("Result7", 0);
	// imshow("Result7", dataScore);
	// resizeWindow("Result7", dataScore.cols, dataScore.rows);

	return dataScore;
}

void convert2Score()
{
	int RH_cnt = 0;
	int LH_cnt = 0;
	int hFlag = 0;
	int octaveFlag = 0;
	Note tmpNote;

	int scoreTmp;

	for (int i = 0; i < yLineCnt / 3; i++)
	{
		RH_cnt = 0;
		LH_cnt = 0;
		hFlag = 0;
		//octaveFlag = 0;
		tmpNote.value = 0;
		tmpNote.tick = 0;

		for (int j = 0; j < realX / 2 - 1; j++)
		{
			if (brailleSet[j + (i * realX / 2)].value == 17 && brailleSet[j + (i * realX / 2) + 1].value == 22)
			{
				hFlag = 1; // RH
				break;
			}

			if (brailleSet[j + (i * realX / 2)].value == 21 && brailleSet[j + (i * realX / 2) + 1].value == 22)
			{
				hFlag = 2; // LH
				break;
			}
		}

		if (hFlag == 1)
		{
			for (int j = 2; j < realX / 2; j++)
			{
				scoreTmp = brailleSet[j + i * (realX / 2)].value;
				
				if (scoreTmp == 16 || scoreTmp == 20 || scoreTmp == 21 || scoreTmp == 4 || scoreTmp == 17 || scoreTmp == 5 || scoreTmp == 1)
				{
					if(scoreTmp == 17)
					{
						if(brailleSet[j + (i * realX / 2) + 1].value != 22)
							octaveFlag = checkOctave(scoreTmp);
					}
					else
						octaveFlag = checkOctave(scoreTmp);
					continue;
				}
					
				else
				{
					
					if (scoreTmp >= 52 && scoreTmp <= 55) // C
					{
						//printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag;
						tmpNote.tick = (int)pow(2, scoreTmp - 49);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp >= 36 && scoreTmp <= 39) // D
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag + 2;
						tmpNote.tick = (int)pow(2, scoreTmp - 33);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp >= 56 && scoreTmp <= 59) // E
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag + 4;
						tmpNote.tick = (int)pow(2, scoreTmp - 53);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp >= 60 && scoreTmp <= 63) // F
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag + 5;
						tmpNote.tick = (int)pow(2, scoreTmp - 57);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp >= 44 && scoreTmp <= 47) // G
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag + 7;
						tmpNote.tick = (int)pow(2, scoreTmp - 41);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp >= 24 && scoreTmp <= 27) // A
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag + 9;
						tmpNote.tick = (int)pow(2, scoreTmp - 21);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp >= 28 && scoreTmp <= 31) // B
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = 12 * octaveFlag + 11;
						tmpNote.tick = (int)pow(2, scoreTmp - 25);
						if (j != realX - 1)
							if (brailleSet[j + (i * realX / 2) + 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp == 50)
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = -1;
						tmpNote.tick = 64;
						if (j != 0)
							if (brailleSet[j + (i * realX / 2) - 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp == 35)
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = -1;
						tmpNote.tick = 32;
						if (j != 0)
							if (brailleSet[j + (i * realX / 2) - 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp == 43)
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = -1;
						tmpNote.tick = 16;
						if (j != 0)
							if (brailleSet[j + (i * realX / 2) - 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else if (scoreTmp == 51)
					{
						// printf("//%d\n", octaveFlag);
						tmpNote.value = -1;
						tmpNote.tick = 8;
						if (j != 0)
							if (brailleSet[j + (i * realX / 2) - 1].value == 2) // dot
								tmpNote.tick += tmpNote.tick / 2;
					}
					else
						continue;

					noteSet.push_back(tmpNote);
				}
			}
		}
		else if (hFlag == 2)
		{
			continue;
		}
		else
			continue;

		// tmpNote.value = -2;
		// tmpNote.tick = -2;
		// noteSet.push_back(tmpNote);
	}
}

int checkOctave(int octave)
{
	switch (octave)
	{
	case 16:
		octave= 1;
		break;
	case 20:
		octave = 2;
		break;
	case 21:
		octave = 3;
		break;
	case 4:
		octave = 4;
		break;
	case 17:
		octave = 5;
		break;
	case 5:
		octave = 6;
		break;
	case 1:
		octave = 7;
		break;
	}
	return octave;
}