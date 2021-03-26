#include "system.h"

#include "cuda_runtime.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>

#include "tracking/device_image.h"
#include "utils/safe_call.h"

// #define CUDA_MEM
// #define TIMING
// #define LOG

namespace fusion
{

System::~System()
{
    // graph->terminate();
    // graphThread.join();
    // delete detector;
}

System::System(bool bSemantic, bool bLoadDiskMap)
    : frame_id(0), reloc_frame_id(0), frame_start_reloc_id(0), is_initialized(false), hasNewKeyFrame(false), b_reloc_attp(false)
{
    safe_call(cudaGetLastError());
    odometry = std::make_shared<DenseOdometry>();

    #ifdef CUDA_MEM
        // inaccurate, the driver decides when to release the memory
        size_t free_t0, total_t0;
        float free_1, free_2, total_0, used_0;
        cudaMemGetInfo(&free_t0, &total_t0);
        total_0 = (uint)total_t0/1048576.0 ;
        free_1 = (uint)free_t0/1048576.0 ;
    #endif
    manager = std::make_shared<SubmapManager>();
    if(bLoadDiskMap)
        manager->readMapFromDisk();
    else
        manager->Create(0, true, true);
    // manager->SetTracker(odometry);
    odometry->SetManager(manager);
    #ifdef CUDA_MEM
        cudaMemGetInfo(&free_t0, &total_t0);
        free_2 = (uint)free_t0/1048576.0 ;
        used_0 = free_1 - free_2;
        std::cout << "## Create a new submap used " << used_0 << " MB memory." << std::endl
                << "   with " 
                << free_1 << " MB free mem before, "
                << free_2 << " MB free mem after" << std::endl 
                << "   out of " << total_0 << " MB total memroy." << std::endl;
    #endif

    /* Semantic & Reloc disabled for now.
    relocalizer = std::make_shared<Relocalizer>(base);
    
    if(bSemantic){
    #ifdef CUDA_MEM
        size_t free_t, total_t;
        float free_m1, free_m2, free_m3, total_m, used_m;
        cudaMemGetInfo(&free_t, &total_t);
        total_m = (uint)total_t/1048576.0 ;
        free_m1 = (uint)free_t/1048576.0 ;

        std::clock_t start = std::clock();
    #endif  
    // detector = new semantic::MaskRCNN("bridge");
    detector = new semantic::NOCS("bridge");
    #ifdef CUDA_MEM
        cudaMemGetInfo(&free_t, &total_t);
        free_m2 = (uint)free_t/1048576.0 ;
        used_m = free_m1 - free_m2;
        std::cout << "## Initialize the detector used " << used_m << " MB memory." << std::endl
                << "   with " 
                << free_m1 << " MB free mem before, "
                << free_m2 << " MB free mem after" << std::endl 
                << "   out of " << total_m << " MB total memroy." << std::endl;
        std::cout << "   and takes "
                  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                  << " seconds" << std::endl;
        start = std::clock();
    #endif
    // detector->initializeDetector("/home/lk18493/github/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml", 0);
    detector->initializeDetector("/home/yohann/NNs/NOCS_CVPR2019/logs/nocs_rcnn_res50_bin32.h5", 1);
    #ifdef CUDA_MEM 
        cudaMemGetInfo(&free_t, &total_t);
        free_m3 = (uint)free_t/1048576.0 ;
        used_m = free_m2 - free_m3;
        std::cout << "## Load the model used " << used_m << " MB memory." << std::endl
                << "   with " 
                << free_m2 << " MB free mem before, "
                << free_m3 << " MB free mem after" << std::endl 
                << "   out of " << total_m << " MB total memroy." << std::endl;
        std::cout << "   and takes "
                  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                  << " seconds" << std::endl;
    #endif
    // odometry->SetDetector(detector);
    }

    // LOAD SEMANTIC MAPS
    if(bLoadSMap){
        manager->readSMapFromDisk("map");
    }
    output_file_name = "reloc";
    pause_window = false;
    */
}

void System::initialization()
{
    current_frame->pose = initialPose;
    
    /* Semantic & Reloc diasbled for now
    // set key frame
    current_keyframe = current_frame;
    // graph->add_keyframe(current_keyframe);
    hasNewKeyFrame = true;
    std::cout << "\n-- KeyFrame needed at frame " << current_frame->id << std::endl; 
    */

    is_initialized = true;

#ifdef LOG
    // start a new log file
    std::string name_log = "/home/yohann/SLAMs/object-guided-relocalisation/"+output_file_name+"_log.txt";
    std::ofstream log_file;
    log_file.open(name_log, std::ios::out);
    if(log_file.is_open())
    {
        log_file << "\n";
    }
    else
    {
        std::cout << "!!!!ERROR: Unable to open the pose file." << std::endl;
    }
    log_file.close();
#endif
    // // start a new pose file
    // std::string name_pose = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/CENT/"+output_file_name+".txt";
    // std::ofstream pose_file;
    // pose_file.open(name_pose, std::ios::out);
    // if(pose_file.is_open())
    // {
    //     pose_file << "";
    // }
    // else
    // {
    //     std::cout << "!!!!ERROR: Unable to open the pose file." << std::endl;
    // }
    // pose_file.close();
}

void System::process_images(const cv::Mat depth, const cv::Mat image, 
                            bool bSubmapping, bool bSemantic, bool bRecordSequence)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);
    float max_perct = 0.;
    int max_perct_idx = -1;
    float thres_new_sm = 0.50;
    float thres_passive = 0.20;
    renderIdx = manager->renderIdx;

    // if (bRecordSequence)
    // {
    //     std::string dir = "/home/yohann/SLAMs/datasets/sequence/";
    //     // depth
    //     std::string name_depth = dir + "depth/" + std::to_string(frame_id) + ".png";
    //     cv::imwrite(name_depth, depth);
    //     // color
    //     cv::Mat color;
    //     cv::cvtColor(image, color, CV_RGB2BGR);
    //     std::string name_color = dir + "color/" + std::to_string(frame_id) + ".png";
    //     cv::imwrite(name_color, color);
    // }

    std::cout << "Frame #" << frame_id << std::endl;
    // In tracking and Mapping, loop through all active submaps
    for(size_t i=0; i<manager->vActiveSubmaps.size(); ++i)
    {
        odometry->setSubmapIdx(i);

        // NOTE new pointer created here!
        // new frame for every submap
        current_frame = std::make_shared<RgbdFrame>(depth_float, image, frame_id, 0);

        /* INITIALIZATION */ 
        if (!is_initialized)
            initialization();

        /* TRACKING */
        if (!odometry->trackingLost){
            b_reloc_attp = false;
            std::cout << "- Dense Tracking." <<  std::endl;
            // update pose of current_frame and reference_frame in corresponding DeviceImage
            odometry->trackFrame(current_frame);
            
            /* Semantic & Reloc diasbled for now
            //---- only create kf in the rendering map ----
            if(keyframe_needed() && i == renderIdx && !odometry->trackingLost)
            {
                create_keyframe();
            }
            */
        }

        /* RENDERING */
        if (!odometry->trackingLost)
        {
            /* Attepmt to not use device image in tracking. Result in noisy reconstruction.
            // cv::cuda::GpuMat cuImage, cuDepth, cuVMap;
            // odometry->get_current_color().copyTo(cuImage);
            // odometry->get_current_depth().copyTo(cuDepth);
            // odometry->get_current_vmap().copyTo(cuVMap);
            
            // Sophus::SE3d Tcm = current_frame->pose; // transformation from camera to map
            // std::cout << "Pose used for fusing and raytracing:\n"
            //           << Tcm.matrix() << std::endl;

            // // update the map
            // manager->active_submaps[i]->update(cuDepth, cuImage, Tcm);
            
            // // ray trace;
            // // cv::Mat test_img, test_vmap, test_nmap;
            // // cuImage.download(test_img);
            // // cv::cvtColor(test_img, test_img, CV_RGB2BGR);
            // // cuVMap.download(test_vmap);
            // // cuNMap.download(test_nmap);
            // manager->active_submaps[i]->raycast(cuVMap, cuImage, Tcm);
            // // cv::Mat test_raycasted_img, test_raycasted_vmap;
            // // cuImage.download(test_raycasted_img);
            // // cv::cvtColor(test_raycasted_img, test_raycasted_img, CV_RGB2BGR);
            // // cuVMap.download(test_raycasted_vmap);
            // // cv::imshow("nmap before raycast", test_nmap);
            // // cv::imshow("vmap before raycast", test_vmap);
            // // // cv::imshow("img before raycast", test_img);
            // // cv::imshow("vmap after raycast", test_raycasted_vmap);
            // // // cv::imshow("img after raycast", test_raycasted_img);
            // // cv::waitKey(0);

            // std::cout << "Update tracker references." << std::endl;
            // odometry->update_reference_model(cuVMap); // update vmap & nmap in tracker
            */

            auto current_image = odometry->get_current_image();
            auto current_frame = current_image->get_reference_frame();
            
            cv::cuda::GpuMat cuDepth, cuVMap; // cuImage, cuNMap;
            // current_image->get_image().copyTo(cuImage);
            current_image->get_raw_depth().copyTo(cuDepth);
            current_image->get_vmap().copyTo(cuVMap);
            // current_image->get_nmap(0).copyTo(cuNMap);
            Sophus::SE3d Tcm = current_frame->pose; // transformation from camera to map
            std::cout << Tcm.matrix() << std::endl;
            
            // update the map
            std::cout << "- Map Fusing." << std::endl;
            manager->vActiveSubmaps[i]->Fuse(cuDepth, Tcm);

            std::cout << "- Raytracing." << std::endl;
            // cv::Mat test_vmap, test_nmap;
            // cuVMap.download(test_vmap);
            // // cuNMap.download(test_nmap);
            manager->vActiveSubmaps[i]->RayTrace(Tcm);
            cuVMap = manager->vActiveSubmaps[i]->GetRayTracingResult();
            
            auto reference_image = odometry->get_reference_image(i);
            reference_image->resize_device_map(cuVMap); 
            // cv::Mat test_raycasted_vmap;
            // cuVMap.download(test_raycasted_vmap);
            // // cv::imshow("nmap before raycast", test_nmap);
            // cv::imshow("vmap before raycast", test_vmap);
            // cv::imshow("vmap after raycast", test_raycasted_vmap);
            // cv::waitKey(0);

            // auto reference_image = odometry->get_reference_image(i);
            // auto reference_frame = reference_image->get_reference_frame();
            // // cv::cuda::GpuMat cuImage, cuDepth, cuVMap, cuNMap;
            // cv::cuda::GpuMat cuImage = reference_image->get_image();
            // cv::cuda::GpuMat cuDepth = reference_image->get_raw_depth();
            // Sophus::SE3d Tcm = reference_frame->pose; // transformation from camera to map
            // std::cout << "Pose used for fusing and raytracing:\n"
            //           << Tcm.matrix() << std::endl;
            // // update the map
            // std::cout << "Map fusing" << std::endl;
            // manager->active_submaps[i]->update(cuDepth, cuImage, Tcm);
            // std::cout << "Raytracing " << std::endl;
            // cv::cuda::GpuMat cuVMap = reference_image->get_vmap();
            // cv::cuda::GpuMat cuNMap = reference_image->get_nmap(0);
            // manager->active_submaps[i]->raycast(cuVMap, cuImage, Tcm);
            // reference_image->resize_device_map(cuVMap); 

            
            // if(manager->active_submaps[i]->bRender){
            //     // update the map
            //     std::cout << "Map fusing" << std::endl;
            //     manager->active_submaps[i]->update(reference_image);
            //     std::cout << "Raytracing " << std::endl;
            //     manager->active_submaps[i]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
            //     reference_image->resize_device_map(); 
            //     /*Semantic & Reloc diasbled for now
            //     // add new keyframe in the map & calculate cuboids for objects detected
            //     if(hasNewKeyFrame && bSemantic){
            //         manager->AddKeyFrame(current_keyframe);
            //         reference_image->downloadVNM(odometry->vModelFrames[i], false);
            //         // // Store vertex map.
            //         // // Store vertex map of current KF as point cloud in the file.
            //         // // std::cout << "Type of vmap is ";
            //         // // std::cout << odometry->vModelFrames[i]->vmap.type() << std::endl;
            //         // std::ofstream pcd_file;
            //         // std::string pcd_file_name = "point_cloud_bin_" + std::to_string(frame_id) + ".txt";
            //         // pcd_file.open(pcd_file_name, std::ios::app);
            //         // int channel = odometry->vModelFrames[i]->vmap.channels();
            //         // int rows = odometry->vModelFrames[i]->vmap.rows;
            //         // int cols = odometry->vModelFrames[i]->vmap.cols;
            //         // float* vmap_data = (float*) odometry->vModelFrames[i]->vmap.data;
            //         // if(pcd_file.is_open())
            //         // {
            //         //     for(int vmapi=0; vmapi < rows*cols; ++vmapi){
            //         //         pcd_file << vmap_data[vmapi*4] << "," << vmap_data[vmapi*4+1] << "," 
            //         //              << vmap_data[vmapi*4+2] << "," << vmap_data[vmapi*4+3] << "\n";
            //         //     } 
            //         // }
            //         // pcd_file.close();
            //         // perform semantic analysis on keyframe
            //         extract_semantics(odometry->vModelFrames[i], false, 1, 0.002, 5, 7);
            //         if(current_keyframe->numDetection > 0){
            //             manager->active_submaps[i]->update_objects(odometry->vModelFrames[i]);
            //             // INTEROGATIVE: Do we actually need this step?
            //             manager->active_submaps[i]->color_objects(reference_image);
            //             manager->active_submaps[i]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_image->get_object_mask(), reference_frame->pose);
            //         }
            //     }
            //     */
            // } else {
            //     manager->active_submaps[i]->check_visibility(reference_image);
            //     manager->active_submaps[i]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
            //     reference_image->resize_device_map();
            // }
            // set current map idx to trackIdx in the odometry
            odometry->setTrackIdx(i);
        }

        /* RELOCALIZATION */
        else
        {
            std::cout << "Relocalisation disabled for now." << std::endl;
            return;

            // std::cout << "\n !!!! Tracking Lost at frame " << frame_id << "! Trying to recover..." << std::endl;
            // if(!bSemantic)
            //     return;
            // reloc_frame_id++;
            // relocalization();
        }

        // if(bSubmapping)
        // {
        //     // check visible block percentage
        //     float tmp_perct = manager->CheckVisPercent(i);
        //     if(tmp_perct < thres_passive){
        //         std::cout << "Move submap " << manager->active_submaps[i]->submapIdx << " from active to passive."
        //                   << " With visible_percentage = " << tmp_perct << std::endl;
        //         manager->activeTOpassiveIdx.push_back(i);
        //     }
        //     if(tmp_perct > max_perct){
        //         max_perct = tmp_perct;
        //         max_perct_idx = i;
        //     }
        // }
    } // end for-active_submaps

    /* POST-PROCESSING */
    // if(bSubmapping)
    // {
    //     // deactivate unwanted submaps
    //     if(manager->activeTOpassiveIdx.size()>0)
    //     {
    //         renderIdx -= manager->activeTOpassiveIdx.size();
    //         manager->CheckActive();
    //     }
    //     // std::cout << "check new sm/render&track" << std::endl;
    //     // check if new submap is needed
    //     if(max_perct < thres_new_sm)
    //     {
    //         std::cout << "NEW SUBMAP NEEDED at frame " << current_frame->id << std::endl;
    //         // int new_map_idx_all = manager->all_submaps.size();
    //         int new_map_idx_all = manager->active_submaps.size() + manager->passive_submaps.size();
    //         manager->Create(base, new_map_idx_all, odometry->vModelDeviceMapPyramid[renderIdx], 
    //                         false, true);
    //         renderIdx = manager->renderIdx;
    //         // create_keyframe();
    //         odometry->vModelDeviceMapPyramid[renderIdx]->downloadVNM(odometry->vModelFrames[renderIdx], odometry->trackingLost);
    //         manager->AddKeyFrame(current_keyframe);
    //     } 
    //     // check which submap to track and render
    //     else
    //     {
    //         // std::cout << renderIdx << "-" << odometry->vModelFrames.size() << std::endl;
    //         manager->CheckTrackAndRender(odometry->vModelFrames[renderIdx]->id, max_perct_idx);
    //     }
    // }

    /* OPTIMIZATION */
    // if (hasNewKeyFrame)
    // {
    //     hasNewKeyFrame = false;
    // }

    // if (bRecordSequence)
    // {
    //     std::string dir = "/home/yohann/SLAMs/datasets/sequence/";
    //     // pose
    //     auto pose = odometry->vModelFrames[renderIdx]->pose.cast<float>().matrix();
    //     Eigen::Matrix3f rot = pose.topLeftCorner<3,3>();
    //     Eigen::Vector3f trans = pose.topRightCorner<3,1>();
    //     Eigen::Quaternionf quat(rot);
    //     // std::cout << rot << std::endl << trans << std::endl;
    //     // std::cout << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << std::endl;
    //     std::string name_pose = dir + "pose.txt";
    //     std::ofstream pose_file;
    //     pose_file.open(name_pose, std::ios::app);
    //     if(pose_file.is_open())
    //     {
    //         pose_file << odometry->vModelFrames[renderIdx]->id << " " 
    //                   << trans(0) << " " << trans(1) << " " << trans(2) << " "
    //                   << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
    //                   << "\n";
    //     }
    //     else
    //     {
    //         std::cout << "!!!!ERROR: Unable to open the pose file." << std::endl;
    //     }
    //     pose_file.close();
    // }

    frame_id += 1;
    std::cout << "FINISHED current frame.\n" << std::endl;
}


// system controls
void System::change_colour_mode(int colour_mode)
{
    std::cout << "To be implemented." << std::endl;
}
void System::change_run_mode(int run_mode)
{
    std::cout << "To be implemented. ";
    std::cout << "(Switch between SLAM and pure (re)Localisation)";
    std::cout << std::endl;
}
void System::restart()
{
    std::cout << "To be implemented." << std::endl;
    // initialPose = last_tracked_frame->pose;
    is_initialized = false;
    frame_id = 0;

    manager->ResetSubmaps();
    odometry->reset();
    // graph->reset();
}
void System::setLost(bool lost)
{
    odometry->trackingLost = true;
}


// visualization
Eigen::Matrix4f System::get_camera_pose() const
{
    // Eigen::Matrix4f Tmf, Twm; 
    Eigen::Matrix4f T;
    if (odometry->get_reference_image(renderIdx))
    {
        // final display map is primary submap centered.
        T = odometry->get_reference_image(renderIdx)->get_reference_frame()->pose.cast<float>().matrix();
        // Twm = manager->active_submaps[renderIdx]->poseGlobal.cast<float>().matrix();
        // T = Twm * Tmf;
    }
    return T;
}
std::vector<MapStruct *> System::get_dense_maps()
{
    return manager->getDenseMaps();
}


// save and read maps
void System::save_mesh_to_file(const char *str)
{
    // SavePLY();
}
void System::writeMapToDisk() const
{
    manager->writeMapToDisk();
}
void System::readMapFromDisk()
{
    std::cout << "Reading map from disk..." << std::endl;
    manager->readMapFromDisk();
}


// size_t System::fetch_mesh_vertex_only(float *vertex)
// {
//     // return manager->active_submaps[renderIdx]->fetch_mesh_vertex_only(vertex);
// }
// size_t System::fetch_mesh_with_normal(float *vertex, float *normal)
// {
//     // return manager->active_submaps[renderIdx]->fetch_mesh_with_normal(vertex, normal);
// }
// size_t System::fetch_mesh_with_colour(float *vertex, unsigned char *colour)
// {
//     // return manager->active_submaps[renderIdx]->fetch_mesh_with_colour(vertex, colour);
// }
// void System::fetch_key_points(float *points, size_t &count, size_t max)
// {
//     manager->GetPoints(points, count, max);
// }
// void System::fetch_key_points_with_normal(float *points, float *normal, size_t &max_size)
// {
// }

/* Semantic & Reloc diasbled for now
void System::relocalize_image(const cv::Mat depth, const cv::Mat image, const fusion::IntrinsicMatrix base)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);
    float max_perct = 0.;
    int max_perct_idx = -1;
    float thres_new_sm = 0.50;
    float thres_passive = 0.20;
    renderIdx = manager->renderIdx;

    // std::cout << "Frame #" << frame_id << std::endl;
    // In tracking and Mapping, loop through all active submaps
    odometry->setSubmapIdx(renderIdx);
    // new frame for every submap
    current_frame = std::make_shared<RgbdFrame>(depth_float, image, frame_id, 0);

    // INITIALIZATION 
    if (!is_initialized)
        initialization();

    // TRACKING
    // track the first frame to initialize, and load map and reloc
    if(frame_id > frame_start_reloc_id)
        odometry->trackingLost = true;
    if (!odometry->trackingLost){
        b_reloc_attp = false;
        // update pose of current_frame and reference_frame in corresponding DeviceImage
        odometry->trackFrame(current_frame);
        if(keyframe_needed() && !odometry->trackingLost)
        {
            create_keyframe();
        }
    }

    // RENDERING
    if (!odometry->trackingLost)
    {
        auto reference_image = odometry->get_reference_image(renderIdx);
        auto reference_frame = reference_image->get_reference_frame();

        if(manager->active_submaps[renderIdx]->bRender){
            // update the map
            manager->active_submaps[renderIdx]->update(reference_image);
            manager->active_submaps[renderIdx]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
            reference_image->resize_device_map();
            // add new keyframe in the map & calculate cuboids for objects detected
            if(hasNewKeyFrame){
                // nocsdetector->performDetection(image);
                manager->AddKeyFrame(current_keyframe);
                reference_image->downloadVNM(odometry->vModelFrames[renderIdx], false);
                // perform semantic analysis on keyframe
                extract_semantics(odometry->vModelFrames[renderIdx], false, 1, 0.002, 5, 7);

                if(current_keyframe->numDetection > 0){
                    manager->active_submaps[renderIdx]->update_objects(odometry->vModelFrames[renderIdx]);
                    manager->active_submaps[renderIdx]->color_objects(reference_image);
                    manager->active_submaps[renderIdx]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_image->get_object_mask(), reference_frame->pose);
                }
            }
        } 
        // set current map idx to trackIdx in the odometry
        odometry->setTrackIdx(renderIdx);
    }
    
    // RELOCALIZATION
    else
    {
        reloc_frame_id = frame_id - frame_start_reloc_id;
        std::cout << "Tracking Lost at frame " << frame_id << "/" << reloc_frame_id << "! Trying to recover..." << std::endl;
        relocalization();
    }

    if (hasNewKeyFrame)
    {
        hasNewKeyFrame = false;
    }

    frame_id += 1;
    // frame_id += 111;
}

void System::relocalization()
{

#ifdef LOG
    std::string log_string = "---- Tracking Lost at frame " + std::to_string(frame_id) + ". Trying to recover...\n";
#endif

    std::clock_t start = std::clock();
    // perform OBJECT based relocalization
    //-step 1: detect obejcts and estimate poses/cuboids in the current frame
    std::cout << "STEP 1: Extract objects." << std::endl;
    auto current_image = odometry->get_current_image();
    current_image->downloadVNM(current_frame, true);
    extract_semantics(current_frame, false, 1, 0.002, 5, 7);
#ifdef LOG
    log_string += " # of detection stored: " + std::to_string(current_frame->numDetection) + "\n";
#endif
    
    //-step 2: solve the absolute orientation (AO) problem (centroid of box and corresponding cuboid)
    std::cout << "STEP 2: Object based relocalisation." << std::endl;
    bool b_recovered = false, 
         b_enough_corresp = false;
    std::vector<std::vector<std::pair<int, std::pair<int, int>>>> vv_inlier_pairs;
    // std::vector<std::vector<std::pair<int, int>>> vv_best_map_cub_labidx;
    std::vector<Eigen::Matrix4d> vtmp_pose_candidates = relocalizer->object_data_association(
                                    current_frame->vObjects, 
                                    manager->active_submaps[renderIdx]->v_objects,
                                    vv_inlier_pairs,
                                    b_enough_corresp, b_recovered
                                );

    if(!b_enough_corresp){
        current_frame->pose = Sophus::SE3d(Eigen::Matrix4d::Identity());
#ifdef LOG
    log_string += " !! Relocalization faild. At least 3 pairs of correspondences needed.\n";
#endif
    }
    else
    {
    //-step 3: use the pose from prob AO as the candidate for icp optimization
        std::cout << "STEP 3: Use icp to validate pose candidates." << std::endl;
        float loss;
        float best_loss = std::numeric_limits<float>::max();
        int recovered_id = -1,
            best_loss_id = -1;
        bool valid_pose = false,
             b_recovered = false;
        int num_of_AO_candidate = vtmp_pose_candidates.size();
#ifdef LOG
    log_string += "In total " + std::to_string(num_of_AO_candidate) + " candidates stored.\n";
    log_string += std::to_string(vv_inlier_pairs[0].size()) + " pairs on inliers are stored.\n";
#endif
        for(size_t pi=0; pi<num_of_AO_candidate; ++pi)
        {
            // Get pose candidate
            Eigen::Matrix4d tmp_pose = vtmp_pose_candidates[pi];
            if(std::isnan(tmp_pose.sum())){
#ifdef LOG
    log_string +=  " !! Invalid pose for candidate " + std::to_string(pi) + "\n";
#endif
                continue;
            }
            valid_pose = true;
            current_frame->pose = Sophus::SE3d(tmp_pose);

            // copy current frame
            RgbdFramePtr current_frame_copy = std::make_shared<RgbdFrame>();
            current_frame->copyTo(current_frame_copy);
            // update cent_matrix with the inlier map cuboid's centroids
            int map_cub_idx = int(pi/2);
            std::cout << "Updating frame cub centroid: " << map_cub_idx << std::endl;
            current_frame->UpdateFrameInliers(vv_inlier_pairs[map_cub_idx]);
            std::cout << "Updated. Reprojecint..." << std::endl;
            current_frame_copy->ReprojectMapInliers(manager->active_submaps[renderIdx]->v_objects,
                                                    vv_inlier_pairs[map_cub_idx]);
            std::cout << "done" << std::endl;
            // current_frame->UpdateFrameInliers(vv_inlier_pairs[map_cub_idx],
            //                                   vv_best_map_cub_labidx[map_cub_idx]);
            // current_frame_copy->ReprojectMapInliers(manager->active_submaps[renderIdx]->v_objects,
            //                                         vv_inlier_pairs[map_cub_idx],
            //                                         vv_best_map_cub_labidx[map_cub_idx]);
            
            // update the referecne/model frame
            std::cout << "Updating reference/model frame" << std::endl;
            odometry->relocUpdate(current_frame_copy);
            auto reference_image = odometry->get_reference_image(renderIdx);
            auto reference_frame = reference_image->get_reference_frame();

            std::cout << "Raycasting" << std::endl;
            // Raycast to update vmap and nmap in the reference/model frame
            manager->active_submaps[renderIdx]->check_visibility(reference_image);
            manager->active_submaps[renderIdx]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
            reference_image->resize_device_map();

            std::cout << "ICPing" << std::endl;
            // depth only ICP and store the pose after icp in the corrresponding position
            // odometry->trackDepthOnly(current_frame, loss);                  // - depth only
            odometry->trackDepthAndCentroid(current_frame, loss);           // - depth + centroid
            Eigen::Matrix4d pose_optimized = current_frame->pose.matrix();
            vtmp_pose_candidates.push_back(pose_optimized);
            std::cout << "ICP done. " << std::endl;

            // Compare
            // both candidates lost, than set to reloc failed
            if(loss < best_loss)
            {
                best_loss = loss;
                best_loss_id = pi;
                if(!odometry->trackingLost){
                    b_recovered = true;
                    recovered_id = vtmp_pose_candidates.size()-1;
                }
            }
#ifdef LOG
    log_string += " Candidate " + std::to_string(pi) + ": ";
    if(b_recovered)
        log_string += "RECOVERED\n";
    else
        log_string += "NOT RECOVERED\n";  
    Eigen::Vector3d tmpTrans = tmp_pose.topRightCorner(3,1);
    Eigen::Vector3d gtTrans = vGTposes[reloc_frame_id-1].cast<double>().topRightCorner(3,1);
    Eigen::Vector3d curTrans = current_frame->pose.matrix().topRightCorner(3,1);
    log_string += "   Translational Diff: " + std::to_string((gtTrans-tmpTrans).norm()) + "\n";
    log_string += "            after icp: " + std::to_string((gtTrans-curTrans).norm());
    log_string += "    icp (Depth+Centroid) loss is " + std::to_string(loss) + "\n";
#endif
        }
        
        std::cout << "STEP 4: Prepare for next tracking." << std::endl;
        if(b_recovered && valid_pose){
            // // TEST the result without depth-centroid icp
            // recovered_id = 0; 
            // std::cout << "---- Best id: " << recovered_id << " out of " << vtmp_pose_candidates.size() << std::endl;
            // std::cout << vtmp_pose_candidates[0] << std::endl;
            // std::cout << vtmp_pose_candidates[1] << std::endl;
            // //////////////////////////////////////////////////

            // update current frame with best pose
            current_frame->pose = Sophus::SE3d( vtmp_pose_candidates[recovered_id] );
            RgbdFramePtr current_frame_copy = std::make_shared<RgbdFrame>();
            current_frame->copyTo(current_frame_copy);
            // upadate reference frame with 
            odometry->relocUpdate(current_frame_copy);
            auto reference_image = odometry->get_reference_image(renderIdx);
            auto reference_frame = reference_image->get_reference_frame();    
            // raycast again for next tracking
            manager->active_submaps[renderIdx]->check_visibility(reference_image);
            manager->active_submaps[renderIdx]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
            reference_image->resize_device_map();

            // set to false in visualization mode
            b_reloc_attp = true;
            // b_reloc_attp = false;
            odometry->trackingLost = false;
            
            // store the relocalized frame as a new KeyFrame
            current_keyframe = current_frame;
            hasNewKeyFrame = true;
#ifdef LOG
    log_string += "Relocalization SUCCEEDED (with " + std::to_string(recovered_id) + ")\n\n";
#endif
            // std::cout << "Relocalization SUCCEEDED (with " + std::to_string(recovered_id) + ")" << std::endl;
        }
        else
        {
            // // TEST the result without depth-centroid icp
            // best_loss_id = 0; 
            // std::cout << "---- Best id: " << best_loss_id << " out of " << vtmp_pose_candidates.size() << std::endl;
            // std::cout << vtmp_pose_candidates[0] << std::endl;
            // std::cout << vtmp_pose_candidates[1] << std::endl;
            // //////////////////////////////////////////////////

            // set to true in visualization mode
            b_reloc_attp = false;
            // b_reloc_attp = true;

            // double check if the candidate is a valid orthogonal matrix
            Eigen::Matrix3d tmpR = vtmp_pose_candidates[best_loss_id].topLeftCorner<3, 3>();
            if( (tmpR*tmpR.transpose()).isIdentity(1e-3) )
                current_frame->pose = Sophus::SE3d( vtmp_pose_candidates[best_loss_id] );
            else
                current_frame->pose = Sophus::SE3d(Eigen::Matrix4d::Identity());
            
            odometry->trackingLost = true;
            hasNewKeyFrame = false;

#ifdef LOG
    log_string += "Relocalization FAILED (with " + std::to_string(best_loss_id) + "\n\n";
#endif
            // std::cout << "Relocalization FAILED (with " + std::to_string(best_loss_id) + "/1)\n";
        }
    } // if enough correspondences

#ifdef LOG
    // log detailed relocalization information
    std::string name_log = "/home/yohann/SLAMs/object-guided-relocalisation/"+output_file_name+"_log.txt";
    std::ofstream log_file;
    log_file.open(name_log, std::ios::app);
    if(log_file.is_open())
    {
        log_file << log_string;
    }
    else
    {
        std::cout << "!!!!ERROR: Unable to open the pose file." << std::endl;
    }
    log_file.close();
#endif
    
    // // store the recovered pose
    // Eigen::Matrix4d tmp_p = current_frame->pose.matrix();
    // Eigen::Matrix3d rot = tmp_p.topLeftCorner<3,3>();
    // Eigen::Vector3d trans = tmp_p.topRightCorner<3,1>();
    // Eigen::Quaterniond quat(rot);
    // std::string name_pose = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/CENT/"+output_file_name+".txt";
    // std::ofstream pose_file;
    // pose_file.open(name_pose, std::ios::app);
    // if(pose_file.is_open())
    // {
    //     pose_file << current_frame->id << " " 
    //             << trans(0) << " " << trans(1) << " " << trans(2) << " "
    //             << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
    //             << "\n";
    // }
    // else
    // {
    //     std::cout << "!!!!ERROR: Unable to open the pose file." << std::endl;
    // }
    // pose_file.close();

    // // // Test with visual
    // // Eigen::Vector3d gtTrans = vGTposes[reloc_frame_id-1].cast<double>().topRightCorner(3,1);
    // // if((gtTrans-trans).norm() > 0.05){
    // //     pause_window = true;
    // // } else {
    // //     pause_window = false;
    // // }

    
    std::cout << "#### Relocalization process takes "
                << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                << " seconds; " << std::endl;
}

bool System::keyframe_needed() const
{
    auto pose = current_frame->pose;
    auto ref_pose = current_keyframe->pose;
    //---- create kf more frequently to get more object detection ----
    // if ((pose.inverse() * ref_pose).translation().norm() > 0.1f)
    if ((pose.inverse() * ref_pose).translation().norm() > 0.05f)
    {
        return true;
    }
    return false;
}

void System::create_keyframe()
{
    current_keyframe = odometry->vModelFrames[renderIdx];
    // graph->add_keyframe(current_keyframe);
    hasNewKeyFrame = true;

    std::cout << "\n-- KeyFrame needed at frame " << odometry->vModelFrames[renderIdx]->id << std::endl; 
}

void System::extract_objects(RgbdFramePtr frame, bool bGeoSeg, float lamb, float tao, int win_size, int thre)
{
    frame->ExtractObjects(detector, false, true, false);   
    if(bGeoSeg){
        cv::Mat edge(frame->row_frame, frame->col_frame, CV_8UC1);
        auto current_keyimage = (!is_initialized || odometry->trackingLost) ? 
                                odometry->get_current_image() : 
                                odometry->get_reference_image(renderIdx);
        current_keyimage->GeometricRefinement(lamb, tao, win_size, edge);
        
        frame->FuseMasks(edge, thre);
    }
    // odometry->vModelDeviceMapPyramid[renderIdx]->upload_semantics(frame);
    odometry->upload_semantics(frame, renderIdx);
    // std::cout << "-- number of objects detected: " << frame->numDetection << std::endl;
}
void System::extract_planes(RgbdFramePtr frame)
{
    // perform k-means on all the background normals
    // first cpu then gpu later
    // at most 3 clusters, x, y, z plane, so start from three and merge later.
    frame->ExtractPlanes();
}
void System::extract_semantics(RgbdFramePtr frame, bool bGeoSeg, float lamb, float tao, int win_size, int thre)
{
    frame->ExtractSemantics(detector, false, true, false);
    // frame->ExtractObjects(detector, false, true, false);
    if(bGeoSeg){
        cv::Mat edge(frame->row_frame, frame->col_frame, CV_8UC1);
        auto current_keyimage = (!is_initialized || odometry->trackingLost) ? 
                                odometry->get_current_image() : 
                                odometry->get_reference_image(renderIdx);
        current_keyimage->GeometricRefinement(lamb, tao, win_size, edge);
        
        frame->FuseMasks(edge, thre);
    }
    // odometry->vModelDeviceMapPyramid[renderIdx]->upload_semantics(frame);
    odometry->upload_semantics(frame, renderIdx);
    // std::cout << "-- number of objects detected: " << frame->numDetection << std::endl;
}

cv::Mat System::get_detected_image()
{
    return current_keyframe->image;
}

cv::Mat System::get_shaded_depth()
{
    if (odometry->get_current_image())
        return cv::Mat(odometry->get_current_image()->get_rendered_image());
}

cv::Mat System::get_rendered_scene() const
{
    return cv::Mat(odometry->get_reference_image(renderIdx)->get_rendered_image());
}

cv::Mat System::get_rendered_scene_textured() const
{
    return cv::Mat(odometry->get_reference_image(renderIdx)->get_rendered_scene_textured());
}

cv::Mat System::get_NOCS_map() const
{
    // CV_32FC3 -> CV_8UC3
    cv::Mat mScaledNOCS(480, 640, CV_8UC3);
    if(!current_keyframe->nocs_map.empty())
        current_keyframe->nocs_map.convertTo(mScaledNOCS, CV_8UC3, 255, 0);
    else
        mScaledNOCS = cv::Mat::zeros(480, 640, CV_8UC3);
    
    return mScaledNOCS;
}

cv::Mat System::get_segmented_mask() const
{
    cv::Mat mScaledMask(480, 640, CV_8UC1);
    if(!current_keyframe->mask.empty())
        current_keyframe->mask.convertTo(mScaledMask, CV_8UC1);
    else
        mScaledMask = cv::Mat::zeros(480, 640, CV_8UC1);
    // return current_keyframe->mEdge;
    return mScaledMask*255;
}

std::vector<Eigen::Matrix<float, 4, 4>> System::getKeyFramePoses() const
{
    return manager->GetKFPoses();
}
std::vector<Eigen::Matrix<float, 4, 4>> System::getGTposes() const
{
    return vGTposes;
}

int System::get_num_objs() const
{
    return manager->GetNumObjs();
    // return current_keyframe->v_cuboids.size();
}
int System::get_reloc_num_objs() const
{
    if(b_reloc_attp){
        return current_keyframe->vObjects.size();
    } else {
        return 0;
    }
}
std::vector<std::pair<int, std::vector<float>>> System::get_objects(bool bMain) const
{
    return manager->GetObjects(bMain);
}
std::vector<std::pair<int, std::vector<float>>> System::get_object_cuboids() const
{
    return manager->GetObjectCuboids();
}
std::vector<std::pair<int, std::vector<float>>> System::get_reloc_cuboids(int usePose) const
{
    // usePose = 1: Ground truth pose
    //           2: pose calculated from centroid
    std::vector<std::pair<int, std::vector<float>>> label_dim_pair;
    if(reloc_frame_id > 0){
        Eigen::Matrix4f Twf;
        switch (usePose)
        {
        case 1:
            Twf = vGTposes[reloc_frame_id-1];
            break;
        case 2:
            Twf = current_frame->pose.cast<float>().matrix();
            break;
        default:
            break;
        }
        // std::cout << current_frame->vObjects.size() << "\n" << Twf << std::endl;
        std::vector<std::shared_ptr<Object3d>>::iterator it;
        for(it=current_frame->vObjects.begin(); it!=current_frame->vObjects.end(); ++it)
        {
            std::vector<float> v_corners(24);
            for(size_t i=0; i<8; ++i)
            {
                Eigen::Vector3f tmp_pt;
                tmp_pt << (*it)->v_all_cuboids[0]->cuboid_corner_pts.at<float>(0,i), 
                          (*it)->v_all_cuboids[0]->cuboid_corner_pts.at<float>(1,i), 
                          (*it)->v_all_cuboids[0]->cuboid_corner_pts.at<float>(2,i);
                tmp_pt = Twf.topLeftCorner(3,3) * tmp_pt + Twf.topRightCorner(3, 1);
                v_corners[3*i] = tmp_pt(0);
                v_corners[3*i+1] = tmp_pt(1);
                v_corners[3*i+2] = tmp_pt(2);
            }
            label_dim_pair.push_back(std::make_pair((*it)->label, v_corners));
        }
    }
	return label_dim_pair;
}

std::vector<float> System::get_obj_centroid_axes(int idx_obj)
{
    return manager->GetObjectCentroidAxes(idx_obj);
}
std::vector<float> System::get_reloc_obj_centroid_axes(int idx_obj, int usePose)
{
    std::vector<float> vAxes;
    // Eigen::Matrix4f Twf;
    // switch (usePose)
    //     {
    //     case 1:
    //         Twf = vGTposes[reloc_frame_id-1];
    //         break;
    //     case 2:
    //         Twf = current_keyframe->pose.cast<float>().matrix();
    //         break;
    //     default:
    //         break;
    //     }
	// for(size_t c=0; c<4; ++c){
    //     Eigen::Vector3f one_pt(current_keyframe->vCuboids[idx_obj]->axes.at<float>(0,c),
    //                            current_keyframe->vCuboids[idx_obj]->axes.at<float>(1,c),
    //                            current_keyframe->vCuboids[idx_obj]->axes.at<float>(2,c));
    //     one_pt = Twf.topLeftCorner(3,3) * one_pt + Twf.topRightCorner(3, 1);
	// 	for(size_t r=0; r<3; ++r){
	// 		vAxes.push_back(one_pt(r));
	// 	}
	// }
	return vAxes;
}

int System::get_object_pts(float *points, size_t &count, int idx_obj)
{
    return manager->GetObjectPts(points, count, idx_obj);
}
int System::get_reloc_obj_pts(float *points, size_t &count, int idx_obj, bool b_useGT)
{
    if(b_reloc_attp){
        count = current_frame->vObjects[idx_obj]->v_all_cuboids[0]->v_data_pts.size();
        Eigen::Matrix4f Twf;
        if(b_useGT){
            Twf = vGTposes[reloc_frame_id-1];
        } else {
            Twf = current_frame->pose.cast<float>().matrix();
        }
        for(size_t i=0; i<count; ++i)
        {
            // camera coordinate frame -> world coordinate frame
            Eigen::Vector3f tmp_pt = Twf.topLeftCorner(3, 3) * 
                                     current_frame->vObjects[idx_obj]->v_all_cuboids[0]->v_data_pts[i] + 
                                     Twf.topRightCorner(3, 1);
            for(size_t j=0; j<3; ++j)
            {
                points[i*3+j] = tmp_pt(j);
            }
        }
        return current_frame->vObjects[idx_obj]->label;
    } else {
        return -1;
    }
}

std::vector<Eigen::Matrix<float, 4, 4>> System::getRelocPoses() const
{
    return vRelocPoses;
}

std::vector<Eigen::Matrix<float, 4, 4>> System::getRelocPosesGT() const
{
    return vRelocPosesGT;
}

void System::load_pose_info(std::string folder, int seq_id){
    std::string name_GT, name_pose, name_geom, name_orb;
    if(seq_id >= 0){
        name_GT = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/GroundTruth/"
                  + folder + "0" + std::to_string(seq_id) + ".txt";
        name_orb = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/ORB/" 
                  + folder + "0" + std::to_string(seq_id) + ".txt";  
    } 
    else {
        std::cout << "!!!!!!! Sequence id SHOULD NOT BE NEGATIVE" << std::endl;
        // name_GT = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/GT_reloc.txt";
        // name_orb = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/ORB_reloc.txt";
    }

    load_from_text(name_GT, vGTposes);
    load_from_text(name_orb, v_NOCSpose_results);

    std::string name_reloc = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/CENT/" 
                             + folder + "0" + std::to_string(seq_id) + ".txt";
    load_from_text(name_reloc, vRelocPoses);
    std::string name_reloc_GT = "/home/yohann/SLAMs/object-guided-relocalisation/pose_info/GroundTruth/" 
                                + folder + "0" + std::to_string(seq_id) + ".txt";
    load_from_text(name_reloc_GT, vRelocPosesGT);
}
void System::load_from_text(std::string file_name, std::vector<Eigen::Matrix4f>& v_results)
{
    v_results.clear();
    std::ifstream pose_file(file_name);
    if(pose_file.is_open())
    {
        std::string one_line;
        while(std::getline(pose_file, one_line))
        {
            std::istringstream ss(one_line);
            double qua[4];
            double trans[3];
            for(size_t i=0; i<8; ++i){
                double one_val;
                ss >> one_val;
                if(i == 0){
                    // std::cout << one_val << std::endl;
                    continue;
                }
                if(i < 4){
                    trans[i-1] = one_val;
                } else {
                    qua[i-4] = one_val;
                }
            }
            Eigen::Quaterniond q(qua);
            Eigen::Vector3d t(trans[0], trans[1], trans[2]);
            // std::cout << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w()
            //           << ", " << t(0) << ", " << t(1) << ", " << t(2) << std::endl;
            // Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.topLeftCorner(3,3) = q.toRotationMatrix();
            T.topRightCorner(3,1) = t;
            v_results.push_back(T.cast<float>());
        }
    }
    else
    {
        std::cout << "NOT FOUND: " << file_name << std::endl;
    }
    pose_file.close();
}

std::vector<Eigen::Matrix4f> System::getNOCSPoseResults() const
{
    return v_NOCSpose_results;
}
std::vector<Eigen::Matrix4f> System::getMaskRCNNResults() const
{
    return v_MaskRCNN_results;
}

std::vector<float> System::get_plane_normals() const
{
    return manager->GetPlaneNormals();
}

// Relocalization with recognitoin
void System::set_frame_id(size_t id)
{
    frame_id = id;
    frame_start_reloc_id = id;
}
*/

// void FuseVoxelMapsAll()
// {
//     auto denseMaps = mpMap->GetDenseMaps();
//     auto denseMapInit = mpMap->GetInitDenseMap();
//     if (!denseMapInit || denseMaps.size() == 0)
//         return;
//     if (denseMapInit->mbInHibernation)
//         denseMapInit->ReActivate();
//     denseMapInit->Reserve(800000, 600000, 800000);
//     denseMapInit->SetActiveFlag(false);
//     for (auto dmap : denseMaps)
//     {
//         if (denseMapInit == dmap || !dmap)
//             continue;
//         if (dmap->mbInHibernation)
//             dmap->ReActivate();
//         denseMapInit->Fuse(dmap);
//         dmap->Release();
//         dmap->mbSubsumed = true;
//         dmap->mpParent = denseMapInit;
//         mpMap->EraseDenseMap(dmap);
//     }
//     denseMapInit->GenerateMesh();
//     denseMapInit->SetActiveFlag(false);
// }

// void SavePLY(const std::string &ply_out)
// {
//     FuseVoxelMapsAll();
//     auto denseMapInit = mpMap->GetInitDenseMap();
//     if (!denseMapInit)
//         return;
//     std::ofstream ply_file(ply_out);
//     ply_file << "ply\n"
//                 << "format ascii 1.0\n"
//                 << "element vertex " << denseMapInit->N / 3 << "\n"
//                 << "property float x\n"
//                 << "property float y\n"
//                 << "property float z\n"
//                 << "element normal " << denseMapInit->N / 3 << "\n"
//                 << "property float x\n"
//                 << "property float y\n"
//                 << "property float z\n"
//                 << "element face " << denseMapInit->N / 9 << "\n"
//                 << "property list uint8 int32 vertex_index\n"
//                 << "end_header\n";
//     // ply_file.write(reinterpret_cast<const char *>(denseMapInit->mplPoint), sizeof(float) * denseMapInit->N);
//     for (int i = 0; i < denseMapInit->N / 3; ++i)
//     {
//         for (int j = 0; j < 3; ++j)
//             ply_file << denseMapInit->mplPoint[i * 3 + j] << " ";
//         ply_file << "\n";
//     }
//     for (int i = 0; i < denseMapInit->N / 3; ++i)
//     {
//         for (int j = 0; j < 3; ++j)
//             ply_file << denseMapInit->mplNormal[i * 3 + j] << " ";
//         ply_file << "\n";
//     }
//     for (int i = 0; i < denseMapInit->N / 9; ++i)
//     {
//         ply_file << "3 ";
//         for (int j = 0; j < 3; ++j)
//             ply_file << i * 3 + j << " ";
//         ply_file << "\n";
//     }
// }

// void SaveTrajectoryFull(std::string abs_path)
// {
//     auto full_trajectory = mpFrameTracker->GetFullTrajectory();
//     auto time_steps = mpFrameTracker->GetTimeStamps();
//     std::ofstream file_out(abs_path);
//     for (int i = 0; i < full_trajectory.size(); ++i)
//     {
//         auto quat = full_trajectory[i].second.unit_quaternion();
//         auto trans = full_trajectory[i].second.translation();
//         file_out << std::fixed
//                     << time_steps[i] << " "
//                     << trans[0] << " "
//                     << trans[1] << " "
//                     << trans[2] << " "
//                     << quat.x() << " "
//                     << quat.y() << " "
//                     << quat.z() << " "
//                     << quat.w() << "\n";
//     }
//     file_out.close();
// }

} // namespace fusion