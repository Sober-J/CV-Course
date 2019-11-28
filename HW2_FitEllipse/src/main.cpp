#include <iostream>

#include <opencv2/opencv.hpp>

#define RED "\033[31m"
#define NORMAL "\033[0m"

cv::Mat img_fitEllipse(const cv::Mat& img, int type);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout<< RED "Please input file name!\n" NORMAL;
        return 1;
    }

    std::string fileName(argv[1]);
    cv::Mat img = cv::imread(fileName);
    if (img.empty())
    {
        std::cout<< RED "File cannot be opened!\n" NORMAL;
        return 1;
    }

    // color -> gray -> binary
    cv::Mat img_gray, img_binary;
	cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_binary, 0, 255, cv::THRESH_OTSU);

    // binary -> contours of ellipses
    cv::Mat img_ellipse = img_fitEllipse(img_binary, img.type());

    cv::Mat img_blend(img.size(), img.type());
    cv::addWeighted(img, 0.4, img_ellipse, 0.6, 0, img_blend);
    cv::imwrite("e_" + fileName, img_blend);
}

cv::Mat img_fitEllipse(const cv::Mat& img, int type)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::Mat img_ellipse = cv::Mat::zeros(img.size(), type);

    for (const auto& it: contours)
    {
        // contours of each ellipse must have more than 6 points
        if (it.size() < 6)
        {
            continue;
        }
        cv::Mat img_points;
        cv::Mat(it).convertTo(img_points, CV_32F);
        cv::RotatedRect box = cv::fitEllipse(img_points);
        cv::ellipse(img_ellipse, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    return img_ellipse;
}