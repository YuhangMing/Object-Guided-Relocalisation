#include <iostream>
#include "system.h"
#include "visualization/main_window.h"
#include "input/azure.h"


int main(int argc, char** argv)
{
    fusion::KinectAzure camera;
    fusion::IntrinsicMatrix K(640, 480, 607.665, 607.516, 321.239, 245.043);
    fusion::System slam(K, 5, false);

    MainWindow window("Semantic_ICP_SLAM", 1920, 920);
    window.SetSystem(&slam);
    cv::Mat image, depth;

    bool bSubmapping = false;
    bool bSemantic = false;
    bool bRecord = true;

    if(bRecord)
    {
        std::ofstream pose_file;
        pose_file.open("/home/yohann/SLAMs/datasets/sequence/pose.txt");
        if(pose_file.is_open())
        {
            pose_file << "# frame_id tx ty tz qx qy qz qw \n";
        }
        else
        {
            std::cout << "!!!!ERROR: Unable to open the pose file." << std::endl;
        }
        pose_file.close();
        window.bRecording = true;
    }

    while (!pangolin::ShouldQuit())
    {
        if (camera.getNextPair(image, depth))
        {
            // std::cout << "Depth: " << depth.type() << " with size " 
            //           << depth.cols << " x " << depth.rows << std::endl;
            // std::cout << "Image: " << image.type() << " with size " 
            //           << image.cols << " x " << image.rows << std::endl;
            window.SetRGBSource(image);
            window.SetDepthSource(depth);   // raw depth
            if (!window.IsPaused())
            {
                slam.process_images(depth, image, K, bSubmapping, bSemantic, bRecord);

                window.SetDetectedSource(slam.get_detected_image());
                // window.SetDepthSource(slam.get_shaded_depth());      // rendered depth
                window.SetRenderScene(slam.get_rendered_scene());
                // window.SetRenderScene(slam.get_rendered_scene_textured());
                // window.SetRenderSceneRaw(slam.get_rendered_scene_raw());
                // window.SetMask(slam.get_segmented_mask());
                window.SetCurrentCamera(slam.get_camera_pose());

                window.mbFlagUpdateMesh = true;
            }

            if (window.IsPaused() && window.mbFlagUpdateMesh)
            {
                // auto *vertex = window.GetMappedVertexBuffer();
                // auto *normal = window.GetMappedNormalBuffer();
                // window.VERTEX_COUNT = slam.fetch_mesh_with_normal(vertex, normal);

                auto *vertex = window.GetMappedVertexBuffer();
                auto *colour = window.GetMappedColourBuffer();
                window.VERTEX_COUNT = slam.fetch_mesh_with_colour(vertex, colour);

                window.mbFlagUpdateMesh = false;
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
