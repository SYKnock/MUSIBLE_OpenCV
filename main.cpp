#include "Braille.h"

int main()
{

    // Mat score = imread("images.jpeg", IMREAD_GRAYSCALE);
    Mat firstScore = imread("./score/score4.jpg", IMREAD_GRAYSCALE);
    Mat score;

    int resizeRow = firstScore.rows * 4 / 5;
    int resizeCols = firstScore.cols * 4 / 5;

    resize(firstScore, score, Size(resizeCols, resizeRow));

    // namedWindow("score", 0);
    // imshow("score", score);
    // resizeWindow("score", resizeCols, resizeRow);

    Mat normalScore = preprocess(score);

    // namedWindow("Result1", 0);
    // imshow("Result1", normalScore);
    // resizeWindow("Result1", resizeCols, resizeRow);

    Mat BrailleScore(score.size(), CV_8UC3);
    BrailleScore = findCircle(normalScore);

    for(int i = 0; i < brailleSet.size(); i++)
        printf("%d ", brailleSet[i].value);
    printf("\n");

    waitKey(0);
    return 0;
}

