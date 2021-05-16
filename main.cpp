#include "Braille.h"

int main(int argc, char *argv[])
{
    // error if there is no input file path
    if (argc < 2)
        return 0;

    int cnt = argc - 1;

    for (int i = 1; i <= cnt; i++)
    {
        xLineCnt = 1;
        yLineCnt = 1;
        realX = 1;
        

        // Mat score = imread("images.jpeg", IMREAD_GRAYSCALE);
        Mat firstScore = imread(argv[i], IMREAD_GRAYSCALE);

        // error if program fail to read image
        if (firstScore.empty())
            return 0;

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

        convert2Score();


        brailleSet.clear();
    }

    for (int i = 0; i < noteSet.size(); i++)
    {
        if (noteSet[i].value == 71)
        {
            printf("%d/%d ", noteSet[i].tick, noteSet[i].value - 12);
            if (i != noteSet.size() - 1)
            {
                if (noteSet[i + 1].value == 69)
                {
                    printf("%d/%d ", noteSet[i + 1].tick, noteSet[i + 1].value - 12);
                    if (i != noteSet.size() - 2)
                    {
                        if (noteSet[i + 2].value == 67)
                        {
                            printf("%d/%d ", noteSet[i + 2].tick, noteSet[i + 2].value - 12);
                            i++;
                        }
                    }
                    i++;
                }
            }
        }
        else
            printf("%d/%d ", noteSet[i].tick, noteSet[i].value);
    }

    printf("\n");

    return 0;
}
