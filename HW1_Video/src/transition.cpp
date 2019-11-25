#include "transition.h"

const int transition_num = 6;

// scaling
void transition_effect0(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_resized;
    cv::Mat img_black(img.size(), img.type());
    for (int i = 1; i <= fps; ++i)
    {  
        img_black.setTo(cv::Scalar(0, 0, 0));
        float rate = (float)i / (float)fps;
        cv::resize(img, img_resized, cv::Size(int(width * rate), int(height * rate)));
        cv::Rect rect((img.cols - img_resized.cols) / 2, (img.rows - img_resized.rows) / 2, img_resized.cols, img_resized.rows);
        img_resized.copyTo(img_black(rect));
        writer<<img_black;
    }
}

// translation from bottom to up
void transition_effect1(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_translation(img.size(), img.type());
    for (int i = 1; i <= fps; ++i)
    {
        img_translation.setTo(cv::Scalar(0, 0, 0));
        cv::Rect rect(0, img.rows * (1 - (float)i / (float)fps), img.cols, img.rows * (float)i / (float)fps);
        img.rowRange(0, img.rows * (float)i / (float)fps).copyTo(img_translation(rect));
        writer<<img_translation;
    }
}

// shutter
void transition_effect2(cv::VideoWriter& writer, const cv::Mat& img)
{
    const int interval = fps * 2;
    cv::Mat img_masked(img.size(), img.type());
    cv::Mat mask(img.size(), img.type());
    for (int i = 0; i < fps; ++i)
    {
        cv::Mat img_tmp;
        int blackWidth = interval - i * 2;
        int whiteWidth = i * 2;
        mask.setTo(cv::Scalar(1.0, 1.0, 1.0));

        for (int j = 0; j < mask.rows / interval; ++j)
        {
            img_tmp = mask.rowRange(j * interval + whiteWidth, (j + 1) * interval);
            img_tmp.setTo(cv::Scalar(0.0,0.0,0.0));
        }
        if ((int)(mask.rows / interval) * interval + whiteWidth < mask.rows)
        {
            img_tmp = mask.rowRange((int)(mask.rows / interval) * interval + whiteWidth, mask.rows);
            img_tmp.setTo(cv::Scalar(0.0,0.0,0.0));
        }

        cv::multiply(img, mask, img_masked);
        writer<<img_masked;
    }
}

// Gaussian blur
void transition_effect3(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_blur;
    for (int i = fps; i > 0; --i)
    {
        cv::GaussianBlur(img, img_blur, cv::Size(2 * i - 1, 2 * i - 1), 0);
        writer<<img_blur;
    }
}

// erasing
void transition_effect4(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_map(img.size(), img.type());
    cv::Mat img_masked(img.size(), img.type());
    cv::Mat mask1(img.size(), img.type());
    cv::Mat mask2(img.size(), img.type());
    cv::Mat img_tmp(img.size(), img.type());
    
    for (int i = 0; i < fps; ++i)
    {
        int interval = (img_map.cols - i * img_map.cols / fps) / 255;
        cv::Mat tmp;
        for (int j = 0; j < 255; ++j)
        {
            tmp = img_map.colRange(j * interval, (j + 1) * interval);
            tmp.setTo(cv::Scalar(j, j, j));
        }
        tmp = img_map.colRange(255 * interval, img_map.cols);
        tmp.setTo(cv::Scalar(255, 255, 255));
        
        mask1.setTo(cv::Scalar(0, 0, 0));
        mask2.setTo(cv::Scalar(1, 1, 1));
        tmp = mask1.colRange(mask1.cols - i * mask1.cols / fps, mask1.cols);
        tmp.setTo(cv::Scalar(1, 1, 1));
        tmp = mask2.colRange(mask2.cols - i * mask2.cols / fps, mask2.cols);
        tmp.setTo(cv::Scalar(0, 0, 0));

        cv::multiply(img, mask1, img_masked);
        cv::multiply(img_map, mask2, img_tmp);
        img_masked = img_masked + img_tmp;
        writer<<img_masked;
    }
}

// binarization
void transition_effect5(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_threshold(img.size(), img.type());
    for (int i = 0; i < fps; ++i)
    {
        threshold(img, img_threshold, i * 255 / fps, 255, cv::THRESH_BINARY);
        writer<<img_threshold;
    }
}

// call six effects in turn
void transition(cv::VideoWriter& writer, const cv::Mat& img)
{
    static transition_effect* transition_array[transition_num] = {  transition_effect0,
                                                                    transition_effect1,
                                                                    transition_effect2, 
                                                                    transition_effect3,
                                                                    transition_effect4,
                                                                    transition_effect5
                                                                    };
    static int i;
    (*(transition_array[i++]))(writer, img);
    if (transition_num == i)
    {
        i = 0;
    }
}