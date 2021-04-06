#include <iostream>
#include "system.h"
#include "input/azure.h"
#include "visualization/main_window.h"
#include "utils/settings.h"

int main(int argc, char** argv)
{
    cv::Mat image, depth;
    SetCalibration();

    fusion::KinectAzure camera;
    fusion::System slam(0);
    std::cout << "\n------------------------------------------------" << std::endl;
    std::cout <<   "-------      Initialised SLAM system     -------" << std::endl;

    MainWindow window("SLAM", 1920, 920);
    window.SetSystem(&slam);
    std::cout <<   "------- Initialised visualisation window -------" << std::endl;
    std::cout <<   "-------      Entering the main loop      -------" << std::endl;
    std::cout <<   "------------------------------------------------" << std::endl;

    while (!pangolin::ShouldQuit())
    {
        if (camera.getNextPair(image, depth))
        {
            // std::cout << "Depth: " << depth.type() << " with size " 
            //           << depth.cols << " x " << depth.rows << std::endl;
            // std::cout << "Image: " << image.type() << " with size " 
            //           << image.cols << " x " << image.rows << std::endl;
            if (!window.IsPaused())
            {
                slam.process_images(depth, image);

                window.SetRGBSource(image);
                window.SetDepthSource(depth);   // raw depth
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;
            }
        }
        window.Render();
    }

    // size_t id = 0;
    // while(id<100){
    // cv::Mat rgb, depth;
    // double time;
    // // rgb in RGB(CV_8UC3), depth in CV_16UC1
    // bool bImages = camera.getNextPair(id, rgb, depth, time);
    // cv::Rect myROI(320, 120, 640, 480);
    // cv::Mat rgb_cropped = rgb(myROI);
    // cv::Mat depth_cropped = depth(myROI);
    // // normalize for the purpose of visualization
    // cv::Mat bgr, scaledDepth;
    // cv::cvtColor(rgb_cropped, bgr, CV_RGB2BGR);
    // normalize(depth_cropped, scaledDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // // scaledDepth.convertTo(scaledDepth, CV_8U, 1);
    // cv::imshow("RGB", bgr);
    // cv::imshow("Depth", scaledDepth);
    // int k = cv::waitKey(0);
    // if(k == 's')
    // {
    //     cv::imwrite("rgb.png", bgr);
    //     cv::imwrite("depth.png", scaledDepth);
    // }
    // std::cout << "Got image? " << bImages << " at " << id << "/" << time << std::endl;
    // }
}
