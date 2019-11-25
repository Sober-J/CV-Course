#include <iostream>
#include <cstring>
#include <vector>
#include <regex>

#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "transition.h"

#define RED "\033[31m"
#define NORMAL "\033[0m"

const int width = 1920, height = 1080;
const int fps = 25;

bool getFiles(std::string& direntName, std::vector<std::string>& jpgFileList, std::string& aviFile);
void writeJPG(cv::VideoWriter& writer, const std::string& direntName, const std::vector<std::string>& jpgFileList);
void writeAVI(cv::VideoWriter& writer, const std::string& direntName, const std::string& aviFile);
void putName(cv::Mat& img);
void makeTitle(cv::VideoWriter& writer);

int main(int argc, char* argv[])
{
    std::ios::sync_with_stdio(false);

    if (argc < 2)
    {
        std::cout<< RED "Please input dirent name!\n" NORMAL;
        return 1;
    }

    std::string direntName(argv[1]);
    std::vector<std::string> jpgFileList;
    std::string aviFile;
    if (!getFiles(direntName, jpgFileList, aviFile))
    {
        return 1;
    }

    cv::VideoWriter writer("video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    std::cout<<"Title\n";
    makeTitle(writer);

    std::cout<<"JPG files:\n";
    writeJPG(writer, direntName, jpgFileList);

    std::cout<<"AVI files:\n";
    writeAVI(writer, direntName, aviFile);

    writer.release();
}

bool getFiles(std::string& direntName, std::vector<std::string>& jpgFileList, std::string& aviFile)
{
    // check whether the dirent exists
    if ('/' != *(direntName.end() - 1))
    {
        direntName.push_back('/');
    }
    if (access(direntName.c_str(), R_OK))
    {
        std::cout<< RED "Dirent does not exist!\n" NORMAL;
        return false;
    }

    // check whether it is a dirent file
    struct stat buf;
    stat(direntName.c_str(), &buf);
    if (!S_ISDIR(buf.st_mode))
    {
        std::cout<< RED "Not a dirent!\n" NORMAL;
        return false;
    }

    std::regex jpgPattern("\\.(jpg|JPG)$");
    std::regex aviPattern("\\.(avi|AVI)$");
    DIR* dp = opendir(direntName.c_str());
    dirent *dirp;
    while ((dirp = readdir(dp)))
    {
        if (DT_REG == dirp->d_type)
        {
            if (std::regex_search(dirp->d_name, jpgPattern))
            {
                jpgFileList.push_back(dirp->d_name);
            }
            else if (std::regex_search(dirp->d_name, aviPattern))
            {
                aviFile = dirp->d_name;
            }
        }
    }
    closedir(dp);

    return true;
}

// draw an OpenCV logo
void makeTitle(cv::VideoWriter& writer)
{
    cv::Mat img(cv::Size(width, height), CV_8UC3);
    putName(img);

    // draw the red part
    for (int i = 0; i <= fps; ++i)
    {
        cv::ellipse(img, cv::Point(width / 2, height / 2 - 160), cv::Size(130, 130), 125, 0, i * 290 / fps, CV_RGB(255, 0, 0), -1);
        cv::circle(img, cv::Point(width / 2, height / 2 - 160), 60, CV_RGB(0, 0, 0), -1);
        writer<<img;
    }
    // draw the green part
    for (int i = 0; i <= fps; ++i)
    {
        cv::ellipse(img, cv::Point(width / 2 - 150, height / 2 + 80), cv::Size(130, 130), 16, 0, i * 290 / fps, CV_RGB(0, 255, 0), -1);
        cv::circle(img, cv::Point(width / 2 - 150, height / 2 + 80), 60, CV_RGB(0, 0, 0), -1);
        writer<<img;
    }
    // draw the blue part
    for (int i = 0; i <= fps; ++i)
    {
        cv::ellipse(img, cv::Point(width / 2 + 150, height / 2 + 80), cv::Size(130, 130), 300, 0, i * 290 / fps, CV_RGB(0, 0, 255), -1);
        cv::circle(img, cv::Point(width / 2 + 150, height / 2 + 80), 60, CV_RGB(0, 0, 0), -1);
        writer<<img;
    }

    for (int i = 0; i < fps / 3; ++i)
    {
        writer<<img;
    }
}

void writeJPG(cv::VideoWriter& writer, const std::string& direntName, const std::vector<std::string>& jpgFileList)
{
    cv::Mat img, imgResized;
    for (const auto& it: jpgFileList)
    {
        std::cout<<direntName + it<<'\n';
        img = cv::imread(direntName + it);
        cv::resize(img, imgResized, cv::Size(width, height));
        putName(imgResized);
        transition(writer, imgResized);
        for (int i = 0; i < fps; ++i)
        {
            writer<<imgResized;
        }
    }
}

void writeAVI(cv::VideoWriter& writer, const std::string& direntName, const std::string& aviFile)
{
    std::cout<<direntName + aviFile<< '\n';
    cv::Mat img, imgResized;
    cv::VideoCapture capture(direntName + aviFile);
    while (true)
    {
        capture>>img;
        if (img.empty())
        {
            break;
        }
        cv::resize(img, imgResized, cv::Size(width, height));
        putName(imgResized);
        writer<<imgResized;
    }
    capture.release();
}

void putName(cv::Mat& img)
{
    cv::putText(img, "Tong Wu 3170104848", cv::Point(80, 1000), cv::FONT_HERSHEY_TRIPLEX, 5.0, cv::Scalar(255, 255, 255));
}