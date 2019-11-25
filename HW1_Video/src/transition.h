#ifndef TRANSITION_H
#define TRANSITION_H

#include <opencv2/opencv.hpp>

extern const int fps;
extern const int width, height;

void transition(cv::VideoWriter& writer, const cv::Mat& img);

typedef void transition_effect(cv::VideoWriter& writer, const cv::Mat& img);

#endif