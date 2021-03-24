#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include "utils/safe_call.h"
#include "data_struct/map_struct.h"
#include "voxel_hashing/map_proc.h"
#include "voxel_hashing/voxel_hashing.h"

namespace fusion
{

std::vector<cv::Scalar> hough_colors = {CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), CV_RGB(255, 255, 0), CV_RGB(255, 0, 255), CV_RGB(0, 255, 255)};

FUSION_HOST void *deviceMalloc(size_t sizeByte)
{
  void *dev_ptr;
  safe_call(cudaMalloc((void **)&dev_ptr, sizeByte));
  return dev_ptr;
}

FUSION_HOST void deviceRelease(void **dev_ptr)
{
  if (*dev_ptr != NULL)
    safe_call(cudaFree(*dev_ptr));

  *dev_ptr = 0;
}

DenseMapping::DenseMapping(const Eigen::Matrix3f &intrinsics, const int width, const int height,
                           int idx, bool bTrack, bool bRender) 
          : submapIdx(idx), bTrack(bTrack), bRender(bRender)
{
  device_map.create();
  zrange_x.create(height / 8, width / 8, CV_32FC1);
  zrange_y.create(height / 8, width / 8, CV_32FC1);
  K = intrinsics;
  KInv = K;
  KInv(0,0) = 1.0/KInv(0,0);
  KInv(1,1) = 1.0/KInv(1,1);
  
  visible_blocks = (HashEntry *)deviceMalloc(sizeof(HashEntry) * device_map.state.num_total_hash_entries_);
  rendering_blocks = (RenderingBlock *)deviceMalloc(sizeof(RenderingBlock) * 100000);

  reset_mapping();
}

DenseMapping::~DenseMapping()
{
  device_map.release();
  deviceRelease((void **)&visible_blocks);
  deviceRelease((void **)&rendering_blocks);
}

// MAIN UPDATE FUNCTION
void DenseMapping::update(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  auto normal = frame->get_nmap();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::update_weighted(
      device_map.map,
      device_map.state,
      depth,
      normal,
      image,
      pose,
      K,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::update(
    cv::cuda::GpuMat depth,
    cv::cuda::GpuMat image,
    const Sophus::SE3d pose)
{
  count_visible_block = 0;

  cuda::update(
      device_map.map,
      device_map.state,
      depth,
      image,
      pose,
      K,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::raycast(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    const Sophus::SE3d pose)
{
  if (count_visible_block == 0)
    return;

  std::cout << "creating rendering blocks" << std::endl;
  cuda::create_rendering_blocks(
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      pose,
      K);

  if (count_rendering_block != 0)
  {

    std::cout << "raycasting with " << count_rendering_block << " rendering block." << std::endl;
    cuda::raycast_with_colour(
        device_map.map,
        device_map.state,
        vmap,
        vmap,
        image,
        zrange_x,
        zrange_y,
        pose,
        KInv);
  }
}
// --Yohann
void DenseMapping::raycast(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    cv::cuda::GpuMat &mask,
    const Sophus::SE3d pose)
{
  if (count_visible_block == 0)
    return;

  cuda::create_rendering_blocks(
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      pose,
      K);

  if (count_rendering_block != 0)
  {
    cuda::raycast_with_object(
        device_map.map,
        device_map.state,
        vmap,
        vmap,
        image,
        mask,
        zrange_x,
        zrange_y,
        pose,
        KInv);
  }
}
void DenseMapping::create_blocks(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  auto normal = frame->get_nmap();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::create_new_block(
      device_map.map,
      device_map.state,
      depth,
      pose,
      K,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::check_visibility(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  // auto normal = frame->get_nmap();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::check_visibility(
      device_map.map,
      device_map.state,
      depth,
      pose,
      K,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);

  // std::cout << "Num of visible block in the relocalized frame is: " << count_visible_block << std::endl;
}

void DenseMapping::color_objects(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  // auto normal = frame->get_nmap();
  auto mask = frame->get_object_mask();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::color_objects(
    device_map.map,
    device_map.state,
    depth,
    image,
    mask,
    pose,
    K,
    flag,
    pos_array,
    visible_blocks,
    count_visible_block);
}

void DenseMapping::raycast_check_visibility(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    const Sophus::SE3d pose)
{
  fusion::cuda::count_visible_entry(
      device_map.map,
      device_map.size,
      height,
      width,
      K,
      pose.inverse(),
      visible_blocks,
      count_visible_block);

  raycast(vmap, image, pose);
}

void DenseMapping::reset_mapping()
{
  device_map.reset();
}

size_t DenseMapping::fetch_mesh_vertex_only(void *vertex)
{
  uint count_triangle = 0;

  cuda::create_mesh_vertex_only(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_normal(void *vertex, void *normal)
{
  uint count_triangle = 0;

  cuda::create_mesh_with_normal(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_colour(void *vertex, void *colour)
{
  uint count_triangle = 0;

  cuda::create_mesh_with_colour(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      colour);

  return (size_t)count_triangle;
}

void DenseMapping::writeMapToDisk(std::string file_name)
{
  // dense map
  MapStruct<false> host_map;
  host_map.create();
  device_map.download(host_map);
  host_map.writeToDisk(file_name);
  host_map.release();
  std::cout << "Dense map wrote to " << file_name << " and "
            << file_name << ".txt" << std::endl;

  // semantic map
  std::ofstream semanticfile;
  semanticfile.open(file_name+"-semantic.txt", std::ios::out);
  if(semanticfile.is_open())
  {
    semanticfile << v_objects.size() << "\n";
  }
  semanticfile.close();
  for(size_t i=0; i<v_objects.size(); ++i)
  {
    v_objects[i]->writeToFile(file_name);
  }
  std::cout << "Dense map wrote to " << file_name << "-semantic.txt" << std::endl;

  // keyframes
  std::ofstream kfsfile;
  kfsfile.open(file_name+"-kfs.txt", std::ios::out);
  if(kfsfile.is_open())
  {
    for(size_t i=0; i<vKFs.size(); ++i)
    {
      Eigen::Matrix4d tmp_kf = vKFs[i]->pose.matrix();
      Eigen::Matrix3d rot = tmp_kf.topLeftCorner<3,3>();
      Eigen::Vector3d trans = tmp_kf.topRightCorner<3,1>();
      Eigen::Quaterniond quat(rot);

      kfsfile << vKFs[i]->id << " "
              << trans(0) << " " << trans(1) << " " << trans(2) << " "
              << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
              << "\n";
    }
  }
  kfsfile.close();
  std::cout << "KeyFrames wrote to " << file_name << "-kfs.txt" << std::endl;
}

void DenseMapping::readMapFromDisk(std::string file_name)
{
  // dense map
  MapStruct<false> host_map;
  host_map.create();
  host_map.readFromDisk(file_name);
  device_map.upload(host_map);
  host_map.release();

  // semantic map
  std::ifstream semanticfile(file_name+"-semantic.txt", std::ios::in);
  int numObj;
  if (semanticfile.is_open())
  {
    std::string sNumObj;
    std::getline(semanticfile, sNumObj);
    numObj = std::stoi(sNumObj);
    std::cout << sNumObj + " objects in the map." << std::endl;
  }
  semanticfile.close();
  v_objects.clear();
  int start_line = 1;
  for(size_t i=0; i<numObj; ++i)
  {
    std::shared_ptr<Object3d> new_object(new Object3d(file_name, start_line));
    v_objects.push_back(std::move(new_object));
    // start_line ++;
  }
  std::cout << "Primary objects list loaded." << std::endl;
  // create back up dic from loaded objects
  object_dictionary.clear();
  for(size_t i=0; i<numObj; ++i){
    std::shared_ptr<Object3d> one_object(new Object3d(v_objects[i]));
    auto search = object_dictionary.find(one_object->label);
    if(search==object_dictionary.end()){
      std::vector<std::shared_ptr<Object3d>> instance_vec;
      instance_vec.push_back(std::move(one_object));
      object_dictionary.insert(std::make_pair(v_objects[i]->label, instance_vec));
    } else {
      search->second.push_back(std::move(one_object));
    }
  } // -i 
  std::cout << "Backup objects dictionary loaded." << std::endl;
  // TEST
  // display observation counts for each instance
  std::map<int, std::vector<std::shared_ptr<Object3d>>>::iterator it;
  for(it=object_dictionary.begin(); it!=object_dictionary.end(); ++it)
  {
    std::cout << "Object " << it->first << " has " 
              << it->second.size() << " instances detected: ";
    for(size_t i=0; i<it->second.size(); ++i)
    {
      std::cout << it->second[i]->observation_count << " (";
      for(size_t j=0; j<it->second[i]->v_all_cuboids.size(); ++j){
        std::cout << it->second[i]->v_all_cuboids[j]->observation << " ";
      }
      std::cout << ") - ";
    } 
    std::cout << std::endl;
  }

  // keyframes
  std::ifstream kfsfile(file_name+"-kfs.txt");
  if(kfsfile.is_open())
  {
    std::string line;
    while(std::getline(kfsfile, line))
    {
      std::istringstream ss(line);
      double trans[3];
      double qua[4];
      for(size_t i=0; i<8; ++i){
        double one_val;
        ss >> one_val;
        if(i == 0){
          int kf_id = one_val;
        }
        else if(i < 4){
          trans[i-1] = one_val;
        } else {
          qua[i-4] = one_val;
        }
      }
      Eigen::Quaterniond q(qua);
      Eigen::Vector3d t(trans[0], trans[1], trans[2]);
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.topLeftCorner(3,3) = q.toRotationMatrix();
      T.topRightCorner(3,1) = t;
      vKFposes.push_back(T.cast<float>());
    }
  }
  kfsfile.close();
}


float DenseMapping::CheckVisPercent() {
  // copy heap_mem_counter_ from device to host
  int *remain = new int[1];
  safe_call(cudaMemcpy(remain, device_map.map.heap_mem_counter_, sizeof(int), cudaMemcpyDeviceToHost));
  int remain_relu = remain[0] > 0 ? remain[0] : 0;

  visible_percent = float(count_visible_block) / (device_map.state.num_total_voxel_blocks_ - remain_relu);
  // std::cout << visible_percent << ": " << count_visible_block << "/" 
  //           << (device_map.state.num_total_voxel_blocks_ - remain_relu) << "/" 
  //           << remain_relu << std::endl;

  return visible_percent;
}

void DenseMapping::update_planes(RgbdFramePtr frame)
{
  if(plane_normals.size() == 0){
    plane_normals = frame->plane_normals;
    std::cout << "  Initialize planes in the map." << std::endl;
    return;
  }

  for(size_t i=0; i<3; ++i){
    // compute the distance, merge the closest ones together
    int tmpIdx;
    float tmpDist = 0;
    float tmpSign;
    for(size_t j=0; j<3; ++j){
      float dist = plane_normals[i].dot(frame->plane_normals[j]);
      float sign = 1;
      if(dist < 0){
        sign = -1;
        dist = sign * dist;
      } 
      if(dist > tmpDist){
        tmpDist = dist;
        tmpIdx = j;
        tmpSign = sign;
      }
    } //-j
    
    plane_normals[i] = (plane_normals[i] + tmpSign*frame->plane_normals[tmpIdx]) / 2;
    plane_normals[i] /= plane_normals[i].norm();
  } //-i
}

void DenseMapping::update_objects(RgbdFramePtr frame)
{
  //- Initialization
  if(v_objects.size() == 0)
  {
    std::cout << "INITIALIZing object lists in the map." << std::endl;
    // initialize the object list with all detected objects in the frame
    v_objects = frame->vObjects;
    // initializing the backup dictionary
    for(size_t i=0; i<v_objects.size(); ++i){
      int label = v_objects[i]->label;
      auto search = object_dictionary.find(label);
      std::shared_ptr<Object3d> one_object(new Object3d(v_objects[i]));
      if(search==object_dictionary.end()){
        // no match found, add new object class
        std::vector<std::shared_ptr<Object3d>> instance_vec;
        instance_vec.push_back(std::move(one_object));
        object_dictionary.insert(std::make_pair(label, instance_vec));  
      } else {
        // match found, store new instance
        search->second.push_back(std::move(one_object));
      }
    } // -i 
    return;
  }

  //- Find matched objects
  int cur_num_map_objs = v_objects.size();
  std::vector<std::shared_ptr<Object3d>>::iterator it_fob;
  for(it_fob=frame->vObjects.begin(); it_fob<frame->vObjects.end(); ++it_fob)
  {
    int label = (*it_fob)->label;
    std::shared_ptr<Cuboid3d> fObjCub = (*it_fob)->v_all_cuboids[0];
    // search objects with same label
    auto search = object_dictionary.find(label);
    if(search == object_dictionary.end())
    {

      std::cout << "Add new object CLASS to the list." << std::endl;
      // // no match found, add new object class
      // // add the new object list
      // std::shared_ptr<Object3d> new_object(new Object3d(fObjCub));
      // v_objects.push_back(std::move(new_object));

      // add new object to the dictionary
      std::vector<std::shared_ptr<Object3d>> instance_vec;
      std::shared_ptr<Object3d> one_object(new Object3d( (*it_fob) ));
      instance_vec.push_back(std::move(one_object));
      object_dictionary.insert(std::make_pair(label, instance_vec));
    
    }
    else
    {
      // match found, update correpondingly
      // use distance between centroids as the first metric
      int obj_idx_best_match = -1;
      Eigen::Vector3d best_diff;
      best_diff << 0, 0, 0;
      double cent_dist_best_match = std::numeric_limits<double>::max();
      // compute distance between new cub and all the main centroid of stored objects
      for(size_t i=0; i<search->second.size(); ++i)
      {
        Eigen::Vector3d diff = search->second[i]->pos - fObjCub->centroid;
        double distance = (diff).norm();
        // choose the closest stored as candidate to update
        if(distance < cent_dist_best_match)
        {
          obj_idx_best_match = i;
          best_diff = diff;
          cent_dist_best_match = distance;
        }
      }
      
      // use unified criteria for both instance level match and BBs level match (overlap volume)
      // compute iou between frame cub and each map cub, find max
      float overlap_ratio = search->second[obj_idx_best_match]->bbox3d_overlap(fObjCub->centroid, fObjCub->dims, fObjCub->scale);
      if(overlap_ratio > 0.1)
      {
        // accept the correspondence and update the cuboid of the matched instance
        search->second[obj_idx_best_match]->update_object(fObjCub);
      } 
      else 
      {
        // the new detection should be a new instance
        // check if the new cuboid is inside primary objects of other classes.
        bool bInsideOtherObject = false;

        // for(size_t i=0; i<v_objects.size(); ++i)
        // {
        //   if(v_objects[i]->label != fObjCub->label){
        //     Eigen::Vector3d tmp_diff = v_objects[i]->pos - fObjCub->centroid;
        //     double distance = (tmp_diff).norm();
        //     if(distance < cent_dist_best_match){
        //       bInsideOtherObject = true;
        //       // CHECK LATER. Instead of discard directly, compare the confidence?
        //       std::cout << "New cuboid inside another object, DISCARDED. " << std::endl;
        //       break;
        //     }
        //   }
        // }

        if(!bInsideOtherObject)
        {
          std::cout << "Add new object INSTANCE to the list." << std::endl;
          // add as new instance in the map
          std::shared_ptr<Object3d> new_object(new Object3d(fObjCub));
          search->second.push_back(std::move(new_object));
        }
      }

      
      // // accept the candidate if the best_dist < max(dims_x, dim_z) of map obj (horizontal)
      // std::shared_ptr<Cuboid3d> one_cuboid(new Cuboid3d(
      //   search->second[obj_idx_best_match]->v_all_cuboids[search->second[obj_idx_best_match]->primary_cuboid_idx]
      // ));
      // double map_dimx = double(one_cuboid->dims[0]*one_cuboid->scale/2),
      //        map_dimz = double(one_cuboid->dims[2]*one_cuboid->scale/2);
      // double frm_dimx = double(fObjCub->dims[0]*fObjCub->scale/2),
      //        frm_dimz = double(fObjCub->dims[2]*fObjCub->scale/2);
      // // map_dimy = double(one_cuboid->dims[1]*one_cuboid->scale/2),
      // // frm_dimy = double(fObjCub->dims[1]*fObjCub->scale/2),
      // double frm_xz = sqrt(frm_dimx*frm_dimx + frm_dimz*frm_dimz);
      // double cent_thre_max = std::max(sqrt(map_dimx*map_dimx + map_dimz*map_dimz), frm_xz),
      //        cent_thre_min = frm_xz/2;
      // double dist_xz = sqrt(best_diff(0)*best_diff(0) + best_diff(2)*best_diff(2));
      // // bool map_in_frm = (std::abs(best_diff(0))<frm_dimx && std::abs(best_diff(1))<frm_dimy && std::abs(best_diff(2))<frm_dimz),
      // //      frm_in_map = (std::abs(best_diff(0))<map_dimx && std::abs(best_diff(1))<map_dimy && std::abs(best_diff(2))<map_dimz);
      // if(dist_xz<cent_thre_max)
      // {
      //   // double check which cuboid it overlaps the most
      //   if(search->second[obj_idx_best_match]->bbox3d_overlap(fObjCub->centroid, fObjCub->dims, fObjCub->scale))
      //   {
      //     // std::cout << "Update existing objects in the map. " << std::endl;
      //     search->second[obj_idx_best_match]->update_object(fObjCub);
      //   }
      //   else
      //   {
      //     // Consider add this cuboid as new instance.
      //     std::cout << "!!! Candidate not overlap with the new cuboid. DISCARD." << std::endl;
      //   }
      // } 
      // else
      // {
      //   // check if the new cuboid is inside primary objects of other classes.
      //   bool bInsideOtherObject = false;
      //   for(size_t i=0; i<v_objects.size(); ++i)
      //   {
      //     if(v_objects[i]->label != fObjCub->label){
      //       Eigen::Vector3d tmp_diff = v_objects[i]->pos - fObjCub->centroid;
      //       double distance = sqrt(tmp_diff(0)*tmp_diff(0) + tmp_diff(2)*tmp_diff(2));
      //       if(distance < cent_thre_min){
      //         bInsideOtherObject = true;
      //         // CHECK LATER. Instead of discard directly, compare the confidence?
      //         std::cout << "New cuboid inside another object, DISCARDED. " << std::endl;
      //         break;
      //       }
      //     }
      //   }
      //   if(!bInsideOtherObject)
      //   {
      //     std::cout << "Add new object INSTANCE to the list." << std::endl;
      //     // add as new instance in the map
      //     std::shared_ptr<Object3d> new_object(new Object3d(fObjCub));
      //     search->second.push_back(std::move(new_object));
      //   }
      // }




    }
  }// -it_fob frame objects 

  //- Update the primary list
  v_objects.clear();
  int obs_thre = vKFs.size()/4;
  for(auto const& it_dic_map_obj : object_dictionary)
  {
    // store all object has more observations than the obs_thre
    for(size_t i=0; i<it_dic_map_obj.second.size(); ++i)
    {
      if(it_dic_map_obj.second[i]->observation_count > obs_thre)
      {
        std::shared_ptr<Object3d> tmp_object(new Object3d(it_dic_map_obj.second[i]));
        v_objects.push_back(std::move(tmp_object));
      }
    }
  }//-it_dic_map_obj

  // //- Update the primary list
  // v_objects.clear();
  // for(auto const& it_dic_map_obj : object_dictionary)
  // {
  //   // find the max observation
  //   int max_observation = 0;
  //   for(size_t i=0; i<it_dic_map_obj.second.size(); ++i)
  //   {
  //     if(it_dic_map_obj.second[i]->observation_count > max_observation)
  //     {
  //       max_observation = it_dic_map_obj.second[i]->observation_count;
  //     }
  //   }
  //   int obs_thre = max_observation/3;
  //   // int obs_thre = 0;

  //   // store all object has more observations than the obs_thre
  //   for(size_t i=0; i<it_dic_map_obj.second.size(); ++i)
  //   {
  //     if(it_dic_map_obj.second[i]->observation_count > obs_thre)
  //     {
  //       std::shared_ptr<Object3d> tmp_object(new Object3d(it_dic_map_obj.second[i]));
  //       v_objects.push_back(std::move(tmp_object));
  //     }
  //   }
  // }//-it_dic_map_obj

  // display observation counts for each instance
  std::map<int, std::vector<std::shared_ptr<Object3d>>>::iterator it;
  std::cout << "Updated map object info:\n";
  for(it=object_dictionary.begin(); it!=object_dictionary.end(); ++it)
  {
    std::cout << "Label " << it->first << " has " 
              << it->second.size() << " instances detected: ";
    for(size_t i=0; i<it->second.size(); ++i)
    {
      std::cout << it->second[i]->observation_count << " (";
      for(size_t j=0; j<it->second[i]->v_all_cuboids.size(); ++j){
        std::cout << it->second[i]->v_all_cuboids[j]->observation << " ";
      }
      std::cout << ") - ";
    } 
    std::cout << std::endl;
  }
}

// void DenseMapping::update_cuboids(RgbdFramePtr frame)
// {
//   // std::cout << "Dictionary_size=" << cuboid_dictionary.size() << ": ";
//   // for(std::map<int,std::vector<std::shared_ptr<Cuboid3d>>>::iterator it = cuboid_dictionary.begin(); it != cuboid_dictionary.end(); ++it){
//   //   std::cout << it->first << "; ";
//   // }
//   // std::cout << std::endl;
//   //- Initialize the cuboid list with cuboids in the frame
//   if(object_cuboids.size() == 0){
//     object_cuboids = frame->vCuboids;
//     // std::cout << "~~ Initialize cuboids in the map." << std::endl;
//     //- Initializing the backup dictionary
//     for(size_t i=0; i<object_cuboids.size(); ++i){
//       // store centroid in the list
//       object_cuboids[i]->vAllCentroids.push_back(frame->vCuboids[i]->centroid);
//       // create new back up object list
//       std::vector<std::shared_ptr<Cuboid3d>> instance_vec;
//       std::shared_ptr<Cuboid3d> frame_cuboid(new Cuboid3d(frame->vCuboids[i]));
//       instance_vec.push_back(std::move(frame_cuboid));
//       cuboid_dictionary.insert(std::make_pair(frame->vCuboids[i]->label, instance_vec));
//     } // -i
//     // std::cout << "~~ Initialize backup dictionary" << std::endl;
//     return;
//   }
//   // // TEST: display current detection result
//   // object_cuboids.clear();
//   // object_cuboids = frame->vCuboids;
//   //- Check the label consistency and update the cuboid
//   for(size_t j=0; j<frame->vCuboids.size(); ++j)
//   {
//     //- Search for matches in the dictionary
//     // std::cout << "## Object with Label=" << frame->vCuboids[j]->label << std::endl;
//     auto search = cuboid_dictionary.find(frame->vCuboids[j]->label);
//     //- Object class exist in the list
//     if(search != cuboid_dictionary.end())
//     {
//       //- Search for matches
//       // std::cout << "~~ Looping through the backup list to find best match" << std::endl;
//       int obs_missed_num = 0;
//       float thre_ratio = 0.5, 
//             max_rMap = 0., 
//             max_rFrame=0.;
//       float max_idx = -1;
//       for(size_t o=0; o<search->second.size(); ++o)
//       {
//         obs_missed_num += search->second[o]->observation;
//         float rMap, rFrame;
//         calculate_overlap(search->second[o]->centroid, search->second[o]->dims, search->second[o]->scale,
//                           frame->vCuboids[j]->centroid, frame->vCuboids[j]->dims, frame->vCuboids[j]->scale,
//                           rMap, rFrame);
//         if(rMap > max_rMap){
//           max_rMap = rMap;
//           max_rFrame = rFrame;
//           max_idx = o;
//         }
//       } //-o
//       // accept only if ratio is over 0.5/map OR 0.5/frame
//       if(max_rMap < thre_ratio && max_rFrame < thre_ratio){
//         max_idx = -1;
//       }
//       //- Update the dictionary and compare confidence
//       float max_confidence = 0;
//       int primary_idx = -1;
//       if(max_idx >= 0)
//       {
//         // std::cout << "   Match found, updating..." << std::endl;
//         update_one_cuboid(search->second[max_idx], frame->vCuboids[j]);
//         // std::cout << "   " << search->second[max_idx]->observation 
//         //           << " - " << search->second[max_idx]->confidence << std::endl;
//         if(search->second[max_idx]->confidence > max_confidence){
//           max_confidence = search->second[max_idx]->confidence;
//           primary_idx = max_idx;
//         }
//         // loop though rest cuboids and update confidence
//         std::vector<int> vIdxToRemove;
//         for(size_t o=0; o<search->second.size(); ++o){
//           if(o != max_idx)
//           {
//             // float ratioM, ratioF;
//             // calculate_overlap(search->second[max_idx]->centroid, search->second[max_idx]->dims, search->second[max_idx]->scale,
//             //                   search->second[o]->centroid, search->second[o]->dims, search->second[o]->scale,
//             //                   ratioM, ratioF);
//             // if(ratioM >= thre_ratio || ratioF >= thre_ratio)
//             // {
//             //   update_one_cuboid(search->second[max_idx], search->second[o], false);
//             //   vIdxToRemove.push_back(o);
//             // }else
//             // {
//               // std::vector<float> miss_conf(7, 1./7.);
//               // search->second[o]->update_confidence(miss_conf);
//               search->second[o]->confidence *= 0.5;
//               // std::cout << "   " << search->second[o]->observation 
//               //           << " - " << search->second[o]->confidence << std::endl;
//               if(search->second[o]->confidence > max_confidence){
//                 max_confidence = search->second[o]->confidence;
//                 primary_idx = o;
//               }
//             // }
//           }
//         } //-o
//         // // remove merged cuboid
//         // if(vIdxToRemove.size()>0){
//         //   std::cout << "++++ num of cuboid to be merged: " << vIdxToRemove.size() << std::endl;
//         //   for(size_t o=vIdxToRemove.size()-1; o>-1; --o){
//         //     search->second.erase( search->second.begin()+vIdxToRemove[o] );
//         //     if(vIdxToRemove[o] < primary_idx){
//         //       primary_idx --;
//         //     }
//         //   }
//         //   std::cout << "done." << std::endl;
//         // }
//       } else {
//         // std::cout << "   Match NOT found, adding to the list..." << std::endl;
//         std::vector<float> miss_conf(7, 1./7.);
//         for(size_t o=0; o<search->second.size(); ++o){
//           // search->second[o]->update_confidence(miss_conf);
//           search->second[o]->confidence *= 0.5;
//           // std::cout << "   " << search->second[o]->observation
//           //           << " - " << search->second[o]->confidence << std::endl;
//           if(search->second[o]->confidence > max_confidence){
//             max_confidence = search->second[o]->confidence;
//             primary_idx = o;
//           }
//         } //-ob
//         // for(size_t ob=0; ob<obs_missed_num; ++ob){
//         //   frame->vCuboids[j]->update_confidence(miss_conf);
//         // }
//         frame->vCuboids[j]->confidence *= std::pow(0.5, obs_missed_num);
//         if(frame->vCuboids[j]->confidence > max_confidence){
//           max_confidence = frame->vCuboids[j]->confidence;
//           primary_idx = search->second.size();
//         }
//         std::shared_ptr<Cuboid3d> frame_cuboid(new Cuboid3d(frame->vCuboids[j]));
//         // std::cout << "   " << frame_cuboid->observation
//         //           << " - " << frame_cuboid->confidence << std::endl;
//         search->second.push_back(std::move(frame_cuboid));
//       }
//       //- Choose the most confident one as map cuboid
//       // std::cout << "-- Choose most confident one in the dictionary as new map cuboid... " << std::endl;
//       if(primary_idx >= 0){
//         for(size_t i=0; i<object_cuboids.size(); ++i){
//           if(object_cuboids[i]->label == frame->vCuboids[j]->label){
//               // std::cout << "   Max = " << max_confidence << std::endl;
//               object_cuboids[i]->copyFrom(search->second[primary_idx]);
//               // add new centroid to the map cuboid
//               object_cuboids[i]->vAllCentroids.push_back(frame->vCuboids[j]->centroid);
//               // // update mean and cov using all centroids
//               // int count = object_cuboids[i]->vAllCentroids.size();
//               // object_cuboids[i]->mean = Eigen::Vector3d::Zero();
//               // for(size_t i=0; i<count; ++i){
//               //   object_cuboids[i]->mean += object_cuboids[i]->vAllCentroids[i];
//               // }
//               // object_cuboids[i]->mean /= count;
//               // double sigX=0, 
//               //       sigY=0, 
//               //       sigZ=0;
//               // object_cuboids[i]->cov = Eigen::Matrix3d::Identity();
//               // for(size_t i=0; i<count; ++i){
//               //   Eigen::Vector3d diff = object_cuboids[i]->vAllCentroids[i] - object_cuboids[i]->mean;
//               //   sigX += diff(0) * diff(0);
//               //   sigY += diff(1) * diff(1);
//               //   sigZ += diff(2) * diff(2);
//               // }
//               // object_cuboids[i]->cov(0,0) = sigX/count;
//               // object_cuboids[i]->cov(1,1) = sigY/count;
//               // object_cuboids[i]->cov(2,2) = sigZ/count;
//           }  
//         } // -i
//       } else {
//         std::cout << "max_id < 0, something went wong!!!" << std::endl;
//       }
//     }
//     //- Object class doesnot exist in the list
//     else
//     {
//       // std::cout << "add new detection" << std::endl;
//       //- Add new detection to the map
//       std::shared_ptr<Cuboid3d> frame_cuboid1(new Cuboid3d(frame->vCuboids[j]));
//       object_cuboids.push_back(std::move(frame_cuboid1));
//       // Add centroid to the map list
//       object_cuboids.back()->vAllCentroids.push_back(frame->vCuboids[j]->centroid);
//       // add cuboid to the backup dictionary
//       std::vector<std::shared_ptr<Cuboid3d>> instance_vec;
//       std::shared_ptr<Cuboid3d> frame_cuboid2(new Cuboid3d(frame->vCuboids[j]));
//       instance_vec.push_back(std::move(frame_cuboid2));
//       cuboid_dictionary.insert(std::make_pair(frame->vCuboids[j]->label, instance_vec));
//       //  std::cout << "   Adding new detection to the backup dictionary." << std::endl;
//     }
//   } //-j
//   // display centroid counts
//   for(size_t i=0; i<object_cuboids.size(); ++i)
//   {
//     std::cout << "Object " << object_cuboids[i]->label << " has " 
//               << object_cuboids[i]->vCentroids.size() << "/"
//               << object_cuboids[i]->vAllCentroids.size() << " centroids stored.";
//     auto search = cuboid_dictionary.find(object_cuboids[i]->label);
//     std::cout << "  with observations: ";
//     for(size_t j=0; j<search->second.size(); ++j)
//     {
//       std::cout << search->second[j]->observation << ", ";
//     }
//     std::cout << std::endl;
//     // std::cout << object_cuboids[i]->mean << std::endl;
//     // std::cout << object_cuboids[i]->centroid << std::endl;
//     // std::cout << object_cuboids[i]->cov.inverse() << std::endl;
//   } // -i
// }

// void DenseMapping::update_one_cuboid(std::shared_ptr<Cuboid3d> map, 
//                                      std::shared_ptr<Cuboid3d> obs,
//                                      bool update_conf)
// {
//   /*
//     update the gaussian probability model
//   */
//   map->vCentroids.push_back(obs->centroid);
//   int count = map->vCentroids.size();
//   map->mean = Eigen::Vector3d::Zero();
//   for(size_t i=0; i<count; ++i){
//     map->mean += map->vCentroids[i];
//   }
//   map->mean /= count;
//   // std::cout << "mean is: " << std::endl << map->mean << std::endl;
//   double sigX=0, 
//          sigY=0, 
//          sigZ=0;
//   map->cov = Eigen::Matrix3d::Identity();
//   for(size_t i=0; i<count; ++i){
//     Eigen::Vector3d diff = map->vCentroids[i] - map->mean;
//     sigX += diff(0) * diff(0);
//     sigY += diff(1) * diff(1);
//     sigZ += diff(2) * diff(2);
//   }
//   map->cov(0,0) = sigX/count;
//   map->cov(1,1) = sigY/count;
//   map->cov(2,2) = sigZ/count;
//   /* 
//     using simple average over all observations to update all the parameters 
//   */
//   double pre_weight = double(map->observation),
//          cur_weight = double(obs->observation),
//          sum_weight = pre_weight + cur_weight;
//   // average tranlsation
//   Eigen::Matrix4d pre_pose = map->pose,
//                   cur_pose = obs->pose;
//   Eigen::Matrix3d pre_rot = pre_pose.topLeftCorner(3, 3),
//                   cur_rot = cur_pose.topLeftCorner(3, 3);
//   Eigen::Vector3d new_trans = (pre_weight * pre_pose.topRightCorner(3,1) + 
//                                cur_weight * cur_pose.topRightCorner(3, 1)) / sum_weight;
//   // average rotation
//   Eigen::Quaterniond pre_quat(pre_rot),
//                       cur_quat(cur_rot);
//   Eigen::Vector4d pre_vQuat(pre_quat.w(), pre_quat.x(), pre_quat.y(), pre_quat.z()),
//                   cur_vQuat(cur_quat.w(), cur_quat.x(), cur_quat.y(), cur_quat.z());
//   Eigen::Matrix4d A = (pre_weight*pre_vQuat*pre_vQuat.transpose() + 
//                         cur_weight*cur_vQuat*cur_vQuat.transpose()) / sum_weight;
//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(A);
//   Eigen::Vector4d new_vQuat = eigensolver.eigenvectors().col(3);
//   Eigen::Quaterniond new_quat(new_vQuat(0), new_vQuat(1), new_vQuat(2), new_vQuat(3));
//   // update pose
//   Eigen::Matrix4d new_pose = Eigen::Matrix4d::Identity();
//   new_pose.topLeftCorner(3, 3) = new_quat.toRotationMatrix();
//   new_pose.topRightCorner(3, 1) = new_trans;
//   map->pose = new_pose;
//   // update scale
//   float new_scale = (pre_weight*map->scale + cur_weight*obs->scale) / sum_weight;
//   map->scale = new_scale;
//   // update dims
//   // Potential Problem: principal axis changes when view-angles changes
//   float max_r = (pre_weight * map->dims[0] + cur_weight * obs->dims[0]) / sum_weight;
//   // float max_r = obs->dims[0];
//   float max_g = (pre_weight * map->dims[1] + cur_weight * obs->dims[1]) / sum_weight;
//   // float max_g = obs->dims[1];
//   float max_b = (pre_weight * map->dims[2] + cur_weight * obs->dims[2]) / sum_weight;
//   // float max_r, max_b;
//   // if(map->confidence < obs->confidence)
//   // {
//   //   max_r = new_scale * obs->dims[0];
//   //   // max_g = new_scale * obs->dims[1];
//   //   max_b = new_scale * obs->dims[2];
//   // } else {
//   //   max_r = new_scale * map->dims[0];
//   //   // max_g = new_scale * map->dims[1];
//   //   max_b = new_scale * map->dims[2];
//   // }
//   map->dims[0] = max_r;
//   map->dims[1] = max_g;
//   map->dims[2] = max_b;
//   max_r *= new_scale;
//   max_g *= new_scale;
//   max_b *= new_scale;
//   // update corners, axes and centroids 
//   Eigen::Matrix<float, 3, 4> xyz_axis;	// in the order of origin, z, y, x
//   xyz_axis << 0, 0, 0, max_r/2,
//               0, 0, max_g/2, 0,
//               0, max_b/2, 0, 0;
//   Eigen::Matrix<float, 3, 8> bbox_3d;
//   bbox_3d << max_r/2, max_r/2, -max_r/2, -max_r/2, max_r/2, max_r/2, -max_r/2, -max_r/2,
//               max_g/2, max_g/2, max_g/2, max_g/2, -max_g/2, -max_g/2, -max_g/2, -max_g/2,
//               max_b/2, -max_b/2, max_b/2, -max_b/2, max_b/2, -max_b/2, max_b/2, -max_b/2;
//   // find cuboid corners and axes in world coordinate system
//   cv::Mat corners(3, 8, CV_32FC1), 
//           axis(3, 4, CV_32FC1);
//   Eigen::Matrix3f Rwn = new_pose.cast<float>().topLeftCorner(3,3);
//   Eigen::Vector3f twn = new_pose.cast<float>().topRightCorner(3,1);
//   for(size_t c=0; c<8; ++c)
//   {
//     if(c<4){
//       // Normalized Coordinate -> Camera Coordinate -> World Coordinate
//       Eigen::Vector3f ax = Rwn * xyz_axis.col(c) + twn;
//       for (size_t r = 0; r < 3; r++)
//       {
//         axis.at<float>(r,c) = ax(r);
//       }
//     }
//     Eigen::Vector3f cor = Rwn * bbox_3d.col(c) + twn;
//     for (size_t r = 0; r < 3; r++)
//     {
//       corners.at<float>(r,c) = cor(r);
//     }
//   }
//   map->cuboid_corner_pts = corners.clone();
//   map->axes = axis.clone();
//   Eigen::Vector3d new_centroid(axis.at<float>(0,0), axis.at<float>(1,0), axis.at<float>(2,0));
//   map->centroid = new_centroid;  
//   // update confidence & obervations
//   // P(X|x1,...,xn) = P(x1|X)...P(x2|X)/normalize_term
//   if(update_conf){
//     map->confidence *= obs->confidence;
//     // map->update_confidence(obs->all_class_confidence);
//   }
//   map->observation += obs->observation;
//   // std::cout << map->label << " - " 
//   //           << map->confidence << " - "
//   //           << map->observation << std::endl;
// }

// void DenseMapping::calculate_overlap(Eigen::Vector3d& cent1, std::vector<float>& dim1, float s1,
//                                       Eigen::Vector3d& cent2, std::vector<float>& dim2, float s2,
//                                       float& rMap, float& rFrame)
// {
//   std::vector<float> sDim1, sDim2;
//   for(size_t i=0; i<3; ++i)
//   {
//     sDim1.push_back(dim1[i]*s1);
//     sDim2.push_back(dim2[i]*s2);
//   }
//   std::vector<float> overlap_min, overlap_max;
//   for(size_t i=0; i<3; ++i)
//   {
//     overlap_min.push_back( std::max(cent1(i)-sDim1[i]/2, cent2(i)-sDim2[i]/2) );
//     overlap_max.push_back( std::min(cent1(i)+sDim1[i]/2, cent2(i)+sDim2[i]/2) );
//     if(overlap_min[i] >= overlap_max[i]){
//       // no overlap
//       rMap = 0.;
//       rFrame = 0.;
//       return;
//     }
//   }
//   float overlap_volume = (overlap_max[0]-overlap_min[0])*(overlap_max[1]-overlap_min[1])*(overlap_max[2]-overlap_min[2]);
//   float map_volume = sDim1[0] * sDim1[1] * sDim1[2];
//   rMap = overlap_volume/map_volume;
//   float frame_volume = sDim2[0] * sDim2[1] * sDim2[2];
//   rFrame = overlap_volume/frame_volume;
//   // std::cout << overlap_volume << ", " << map_volume << ", " << overlap_volume/map_volume << std::endl;
//   // return overlap_volume/map_volume; 
// }

// void DenseMapping::estimate_cuboids(RgbdImagePtr frame)
// {
  // std::cout << "GPU Implementation of cuboids estimation, should NOT be called!" << std::endl;
  // // frame is DeviceImage
  // // auto image = frame->get_image();
  // // auto depth = frame->get_raw_depth();
  // // // auto normal = frame->get_nmap();
  // // auto pose = frame->get_reference_frame()->pose;
  // auto mask = frame->get_object_mask();
  // auto vmap = frame->get_vmap();
  // int num_bbox = v_bbox.size();
  // for(int i=0; i<num_bbox; ++i){
  //   // get 2d box and corresponding label
  //   auto bbox = frame->get_object_bbox(i);
  //   int label = frame->get_object_label(i);
  //   // --- GPU Implementation
  //   // // estimate 3d cuboid
  //   // cuda::estimate_cuboids(
  //   //   vmap,
  //   //   mask,
  //   //   bbox,
  //   //   static_cast<unsigned char>(label),
  //   //   cuboid);
  //   // // store 3d cuboid with label
  //   // cv::Mat h_cuboid(1, 6, CV_32F);
  //   // cuboid.download(h_cuboid);
  //   // std::vector v_cuboid = {h_cuboid.at<float>(0,0), h_cuboid.at<float>(0,1), h_cuboid.at<float>(0,2),
  //   //                         h_cuboid.at<float>(0,3), h_cuboid.at<float>(0,4), h_cuboid.at<float>(0,5)};
  //   // object_cuboids.insert(std::make_pair(lable, v_cuboid));
  // }
  // // cuda::
// }

// void DenseMapping::estimate_cuboids(RgbdFramePtr frame, bool tracking_lost)
// {
  // // int cuboid_size = better_cuboids.size();
  // // frame is RgbdFrame
  // auto mask = frame->mask;  // 8UC1
  // int row = mask.rows;
  // int col = mask.cols;
  // auto vmap = frame->vmap;  // 32FC4
  // auto Tf2w = frame->pose.cast<float>();  // when tracking lost, this pose is set to Identity
  // Eigen::Vector3f point;
  // int num_bbox = frame->numDetection;
  // /*// Hough transform on the entire image
  //   // cv::Mat imgGray, imgCanny, imgHough;
  //   // cv::cvtColor(frame->img_original, imgGray, CV_BGR2GRAY);
  //   // cv::Canny(imgGray, imgCanny, 100, 300, 3);
  //   // cv::imwrite("/home/yohann/canny.png", imgCanny);
  //   // std::vector<cv::Vec2f> imgLines;
  //   // cv::HoughLines(imgCanny, imgLines, 1, CV_PI/180, 120);
  //   // imgHough = frame->img_original.clone();
  //   // for(size_t k=0; k<imgLines.size(); ++k)
  //   //   {
  //   //     float rho = imgLines[k][0];
  //   //     float theta = imgLines[k][1];
  //   //     // std::cout << "line detected at " 
  //   //     //           << "rho = " << rho 
  //   //     //           << ", theta = " << theta << std::endl;
  //   //     double a = cos(theta), b = sin(theta);
  //   //     double x0 = a*rho, y0 = b*rho;
  //   //     // std::cout << a << ", " << b << ", " << x0 << ", " << y0 << std::endl;
  //   //     cv::Point pt1, pt2;
  //   //     pt1.x = cvRound(x0 + 1000*(-b));
  //   //     pt1.y = cvRound(y0 + 1000*(a));
  //   //     pt2.x = cvRound(x0 - 1000*(-b));
  //   //     pt2.y = cvRound(y0 - 1000*(a));
  //   //     // aux.clear();
  //   //     // aux.push_back(pt1);
  //   //     // aux.push_back(pt2);
  //   //     // lineSegments.push_back(aux);
  //   //     // std::cout << "with " 
  //   //     //           << "(" << pt1.x << ", " << pt1.y << ") - " 
  //   //     //           << "(" << pt2.x << ", " << pt2.y << "). " 
  //   //     //           << std::endl;
  //   //     cv::line(imgHough, pt1, pt2, CV_RGB(0, 255, 255), 1, 8);
  //   //   }
  //   //   cv::imwrite("/home/yohann/hough.png", imgHough);
  //   /////////////////////////////////////////////////
  // */
  // for(int i=0; i<num_bbox; ++i){
  //   // get 2d box and corresponding label
  //   auto bbox = frame->v_bbox[i];
  //   int label = frame->vLabels[i];
  //   float confidence = frame->vScores[i];
  //   // get bbox
  //   int x_2d_min, y_2d_min, x_2d_max, y_2d_max;
  //   x_2d_min = int( bbox.at<float>(0,0) );
  //   if(x_2d_min < 0)
  //     x_2d_min = 0;
  //   y_2d_min = int( bbox.at<float>(0,1) );
  //   if(y_2d_min < 0)
  //     y_2d_min = 0;
  //   x_2d_max = int( bbox.at<float>(0,2) + 0.5 );
  //   if(x_2d_max > col)
  //     x_2d_max = col;
  //   y_2d_max = int( bbox.at<float>(0,3) + 0.5 );
  //   if(y_2d_max > row)
  //     y_2d_max = row;
  //   std::shared_ptr<Cuboid3d> one_cuboid(new Cuboid3d(label, confidence));
  //   // find valid map point
  //   for(int r=y_2d_min; r<y_2d_max; ++r){
  //     for(int c=x_2d_min; c<x_2d_max; ++c){
  //       // check label consistency
  //       // if(tracking_lost)
  //       //   std::cout << "--check mask consistency" << std::endl;
  //       if(mask.at<unsigned char>(r,c) != one_cuboid->label)
  //         continue;
  //       // get vertex
  //       cv::Vec4f vertex = vmap.at<cv::Vec4f>(r,c);
  //       float x_3d = vertex[0]/vertex[3];
  //       float y_3d = vertex[1]/vertex[3];
  //       float z_3d = vertex[2]/vertex[3];
  //       // if(tracking_lost)
  //       //   std::cout << "--check vertex value" << std::endl;
  //       if(std::isnan(x_3d) || std::isnan(y_3d) || std::isnan(z_3d))
  //         continue;
  //       // transfer from camera coordinate frame to world coordinate frame
  //       point << x_3d, y_3d, z_3d;
  //       point = Tf2w * point;   
  //       // store valid point in the cuboid
  //       one_cuboid->add_point(point);
  //     } // for-col
  //   } /// for-row
  //   if(!tracking_lost)
  //   {
  //     /* Tracking not Lost */
  //     // find principal axes
  //     // one_cuboid->find_principal_axes();
  //     // find bounding cuboids
  //     bool bCFound = one_cuboid->find_bounding_cuboid();
  //     // if(one_cuboid->find_bounding_cuboid())
  //     //   better_cuboids.push_back(one_cuboid);
  //     // if no suitable results found, continue
  //     if(one_cuboid->data_pts.empty())
  //     {
  //       std::cout << "!! Label " << one_cuboid->label << ": No Valid Cuboid found. !!" << std::endl;
  //       continue;
  //     } // if-empty_cuboids
  //     // check if it is previously detected & if one object is detected as two
  //     bool bDetected = false;
  //     bool bOverlap = false;
  //     std::vector<std::shared_ptr<Cuboid3d>>::iterator it;
  //     std::vector<std::shared_ptr<Cuboid3d>>::iterator it_d;
  //     std::vector<std::shared_ptr<Cuboid3d>>::iterator it_o;
  //     for(it = better_cuboids.begin(); it != better_cuboids.end(); ++it){
  //       // assuming only one object per class
  //       // check if it is previously detected //
  //       if(one_cuboid->label == (*it)->label){
  //         bDetected = true;
  //         it_d = it;
  //         break;
  //       } else {
  //         // check if a object is re-detected //
  //         bOverlap = (*it)->is_overlapped(one_cuboid);
  //         if(bOverlap){
  //           it_o = it;
  //           break;
  //         }
  //       }
  //     }
  //     // separate below from above for the purpose of initialization
  //     if(bDetected){
  //       // refine previous one with new detections
  //       (*it_d)->merge_cuboids(one_cuboid);
  //     } else {
  //       if(bOverlap){
  //         // std::cout << "OVERLAP found" << std::endl;
  //         (*it_o)->update_label(one_cuboid);
  //       } else {
  //         if(bCFound)
  //           better_cuboids.push_back(std::move(one_cuboid));
  //         else
  //           std::cout << "!!!!!!! SOMEHOW no cuboid found !!!!!! CHECK HERE !!!!!!!" << std::endl;
  //       }
  //     }
  //   }
  //   else
  //   {
  //     /* Tracking Lost */
  //     if(!one_cuboid->data_pts.empty())
  //     {
  //       // loop through all available cuboids in the submap
  //       // if same labelled one found -> fit current point cloud to the cuboid
  //       // if same labelled one not found -> use raw depth to find cuboid's dim
  //       std::vector<std::shared_ptr<Cuboid3d>>::iterator it;
  //       std::vector<std::shared_ptr<Cuboid3d>>::iterator it_d;
  //       for(it = better_cuboids.begin(); it != better_cuboids.end(); ++it){
  //         // assuming only one object per class
  //         // check if it is previously detected //
  //         if(one_cuboid->label == (*it)->label){
  //           one_cuboid->fit_cuboid((*it), x_2d_max-x_2d_min);
  //           frame->v_cuboids.push_back(std::move(one_cuboid));
  //           break;
  //         } 
  //         if(it == better_cuboids.end())
  //           std::cout << "No MATCH FOUND for fit_cuboid." << std::endl;
  //         // else {
  //         //   // check if a object is re-detected //
  //         //   bOverlap = (*it)->is_overlapped(one_cuboid);
  //         //   if(bOverlap){
  //         //     it_o = it;
  //         //     break;
  //         //   }
  //         // }
  //       }
  //     }
  //     else
  //     {
  //       std::cout << "No valid map point for detected object." << std::endl;
  //     }
  //   } // tracking lost
  // } // for-numDetection -i
  // // if(!tracking_lost){
  // //   std::cout << "-- number of object cuboids in current map is: " << cuboid_size 
  // //             << "->" << object_cuboids.size() << std::endl;
  // //   for(size_t i=0; i<object_cuboids.size(); ++i){
  // //     std::cout << "   object " << object_cuboids[i]->label << " with centroid (" 
  // //               << object_cuboids[i]->centroid[0] << ", "
  // //               << object_cuboids[i]->centroid[1] << ", "
  // //               << object_cuboids[i]->centroid[2] << ")" << std::endl;
  // //   }
  // // }
  // // else {
  // //   std::cout << "-- number of object cuboids in current frame is: " << frame->v_cuboids.size() << std::endl;
  // //   for(size_t i=0; i<frame->v_cuboids.size(); ++i){
  // //     std::cout << "   object " << frame->v_cuboids[i]->label << " with centroid (" 
  // //               << frame->v_cuboids[i]->centroid[0] << ", "
  // //               << frame->v_cuboids[i]->centroid[1] << ", "
  // //               << frame->v_cuboids[i]->centroid[2] << ")" << std::endl;
  // //   } 
  // // }
// }

// void DenseMapping::estimate_quadrics(RgbdFramePtr frame, bool tracking_lost)
// {
  // // frame is RgbdFrame
  // auto mask = frame->mask;  // 8UC1
  // int row = mask.rows;
  // int col = mask.cols;
  // auto vmap = frame->vmap;  // 32FC4
  // auto Tf2w = frame->pose.cast<float>();
  // Eigen::Vector3f point;
  // int num_bbox = frame->numDetection;
  // // loop through every detections
  // for(size_t idx=0; idx < num_bbox; ++idx)
  // {
  //   // get 2d box and corresponding label
  //   auto bbox = frame->v_bbox[idx];
  //   int label = frame->vLabels[idx];
  //   float confidence = frame->vScores[idx];
  //   // bool bDetected = false;
  //   // bool bOverlap = false;
  //   // corner points of the bounding box, in homogeneous coordinates
  //   Eigen::Vector3f topleft(bbox.at<float>(0,0), bbox.at<float>(0,1), 1.), 
  //                   topright(bbox.at<float>(0,0), bbox.at<float>(0,3), 1.), 
  //                   bottomleft(bbox.at<float>(0,2), bbox.at<float>(0,1), 1.), 
  //                   bottomright(bbox.at<float>(0,2), bbox.at<float>(0,3), 1.);  // (x, y)
  //   // std::cout << "(" << topleft(0) << ", " << topleft(1) << "); "
  //   //           << "(" << topright(0) << ", " << topright(1) << "); "
  //   //           << "(" << bottomleft(0) << ", " << bottomleft(1) << "); "
  //   //           << "(" << bottomright(0) << ", " << bottomright(1) << ")" << std::endl;
  //   // from corner points to boundary lines
  //   Eigen::Vector3f top, left, right, bottom;
  //   top = topleft.cross(topright);
  //   left = topleft.cross(bottomleft);
  //   right = topright.cross(bottomright);
  //   bottom = bottomleft.cross(bottomright);
  //   // from boundary lines to plane evelope
  //   // TODO: back projcect the lines to 3D space to form the plane //
  //   // get bbox
  //   int x_2d_min, y_2d_min, x_2d_max, y_2d_max;
  //   x_2d_min = int( bbox.at<float>(0,0) );
  //   if(x_2d_min < 0)
  //     x_2d_min = 0;
  //   y_2d_min = int( bbox.at<float>(0,1) );
  //   if(y_2d_min < 0)
  //     y_2d_min = 0;
  //   x_2d_max = int( bbox.at<float>(0,2) + 0.5 );
  //   if(x_2d_max > col)
  //     x_2d_max = col;
  //   y_2d_max = int( bbox.at<float>(0,3) + 0.5 );
  //   if(y_2d_max > row)
  //     y_2d_max = row;
  //   // construct linear equation matrix
  //   // cv::Mat vec_X;
  //   std::vector<float> vector_X;
  //   std::vector<std::vector<float>> vector_pt_bu;
  //   int count = 0;
  //   for(size_t r=y_2d_min; r<y_2d_max; ++r)
  //   {
  //     for(size_t c=x_2d_min; c<x_2d_max; ++c)
  //     {
  //       // check semantic label
  //       if(mask.at<unsigned char>(r,c) != label)
  //         continue;
  //       // get vertex
  //       cv::Vec4f vertex = vmap.at<cv::Vec4f>(r,c);
  //       float x_3d = vertex[0]/vertex[3];
  //       float y_3d = vertex[1]/vertex[3];
  //       float z_3d = vertex[2]/vertex[3];
  //       // if(tracking_lost)
  //       //   std::cout << "--check vertex value" << std::endl;
  //       if(std::isnan(x_3d) || std::isnan(y_3d) || std::isnan(z_3d))
  //         continue;
  //       // float vXi[10] = {x_3d*x_3d, 2*x_3d*y_3d, 2*x_3d*z_3d, 2*x_3d,
  //       //                             y_3d * y_3d, 2*y_3d*z_3d, 2*y_3d,
  //       //                                          z_3d * z_3d, 2*z_3d,
  //       //                                                       1.    };
  //       // cv::Mat vec_Xi = cv::Mat(1, 10, CV_32F, vXi);
  //       // vec_X.push_back(vec_Xi);
  //       vector_X.push_back(x_3d*x_3d);
  //       vector_X.push_back(2*x_3d*y_3d);
  //       vector_X.push_back(2*x_3d*z_3d);
  //       vector_X.push_back(2*x_3d);
  //       vector_X.push_back(y_3d*y_3d);
  //       vector_X.push_back(2*y_3d*z_3d);
  //       vector_X.push_back(2*y_3d);
  //       vector_X.push_back(z_3d*z_3d);
  //       vector_X.push_back(2*z_3d);
  //       vector_X.push_back(1.);
  //       count++;
  //       std::vector<float> tmp{x_3d, y_3d, z_3d};
  //       vector_pt_bu.push_back(tmp);
  //     }
  //   }
  //   // construct eigen matrix for svd
  //   Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_matX(vector_X.data(), count, 10);
  //   // std::cout << count << ": ";
  //   // std::cout << eigen_matX.rows() << ", " << eigen_matX.cols() << std::endl;
  //   // solve for the best fit quadric
  //   if(eigen_matX.rows() > 0){
  //     // std::clock_t start = std::clock();
  //     // Eigen::JacobiSVD<Eigen::MatrixXf> svd_full(eigen_matX, Eigen::ComputeFullU | Eigen::ComputeFullV);
  //     // std::cout << "#### full svd takes "
  //     //           << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
  //     //           << " seconds; " << std::endl;
  //     // start = std::clock();
  //     Eigen::JacobiSVD<Eigen::MatrixXf> svd_thin(eigen_matX, Eigen::ComputeThinU | Eigen::ComputeThinV);
  //     // std::cout << "#### thin svd takes "
  //     //           << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
  //     //           << " seconds; " << std::endl;
  //     // // std::cout << "Its singular values are:" << std::endl << svd_full.singularValues() << std::endl;
  //     // // std::cout << "Its right singular vectors are the columns of the FULL V matrix:" << std::endl << svd_full.matrixV() << std::endl;
  //     // std::cout << "Its singular values are:" << std::endl << svd_thin.singularValues() << std::endl;
  //     // std::cout << "Its right singular vectors are the columns of the THIN V matrix:" << std::endl << svd_thin.matrixV() << std::endl;
  //     // // Vector3f rhs(1, 0, 0);
  //     // // std::cout << "Now consider this rhs vector:" << std::endl << rhs << std::endl;
  //     // // std::cout << "A least-squares solution of m*x = rhs is:" << std::endl << svd.solve(rhs) << std::endl;
  //     Eigen::Matrix4f Q_;
  //     int v_count = 0;
  //     for(size_t rQ=0; rQ<4; ++rQ){
  //       for(size_t cQ=rQ; cQ<4; ++cQ){
  //         float one_val = svd_thin.matrixV()(v_count, 9);
  //         Q_(rQ, cQ) = one_val;
  //         Q_(cQ, rQ) = one_val;
  //         v_count++;
  //       }
  //     }
  //     std::cout << "--The estimated quadric is:" << std::endl << Q_ << std::endl;
  //     float det = Q_.determinant();
  //     if(det == 0)
  //       std::cout << "!!!Q (coarse) is singular!!! " << std::endl;
  //     std::cout << "  The determinant of Q (coarse) is " << det << std::endl;
  //     // Eigen::Matrix4f Q_dual = det * Q.inverse().transpose();
  //     // std::cout << "The corresponding dual quadric is:" << std::endl << Q_dual << std::endl;
  //     // recover translation parameter
  //     Eigen::Matrix3f Q33 = Q_.topLeftCorner(3,3);
  //     Eigen::Vector3f Q4 = Q_.topRightCorner(3,1);
  //     Eigen::Vector3f trans_v = -1 * Q33.inverse() * Q4;
  //     std::cout << "  The translation parameters are:" << std::endl << trans_v << std::endl;
  //     Eigen::EigenSolver<Eigen::Matrix3f> es_(Q33);
  //     std::cout << "  eigenvalues:" << std::endl;
  //     std::cout << es_.eigenvalues() << std::endl;
  //     std::cout << "  eigenvectors=" << std::endl;
  //     std::cout << es_.eigenvectors() << std::endl;
  //     // solve for origin centered quadric
  //     Eigen::Matrix4f trans_m = Eigen::Matrix4f::Identity();
  //     trans_m.topRightCorner(3,1) = trans_v;
  //     // std::cout << "The translation matrix is:" << std::endl << trans_m << std::endl;
  //     Eigen::Matrix4f trans_m_inv = trans_m.inverse();
  //     // std::cout << "Its inverse is:" << std::endl << trans_m_inv << std::endl;
  //     vector_X.clear();
  //     for(size_t pt_idx = 0; pt_idx < count; ++pt_idx)
  //     {
  //       std::vector<float> one_pt = vector_pt_bu[pt_idx];
  //       float x_3d_trans = vector_pt_bu[pt_idx][0] - trans_v(0);
  //       float y_3d_trans = vector_pt_bu[pt_idx][1] - trans_v(1);
  //       float z_3d_trans = vector_pt_bu[pt_idx][2] - trans_v(2);
  //       vector_X.push_back(x_3d_trans * x_3d_trans);
  //       vector_X.push_back(2 * x_3d_trans * y_3d_trans);
  //       vector_X.push_back(2 * x_3d_trans * z_3d_trans);
  //       vector_X.push_back(2 * x_3d_trans);
  //       vector_X.push_back(y_3d_trans * y_3d_trans);
  //       vector_X.push_back(2 * y_3d_trans * z_3d_trans);
  //       vector_X.push_back(2 * y_3d_trans);
  //       vector_X.push_back(z_3d_trans * z_3d_trans);
  //       vector_X.push_back(2 * z_3d_trans);
  //       vector_X.push_back(1.);
  //     }
  //     // construct eigen matrix for svd
  //     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_matX_trans(vector_X.data(), count, 10);
  //     // std::cout << eigen_matX_trans.rows() << ", " << eigen_matX_trans.cols() << std::endl;
  //     Eigen::JacobiSVD<Eigen::MatrixXf> svd_thin_trans(eigen_matX_trans, Eigen::ComputeThinU | Eigen::ComputeThinV);
  //     Eigen::Matrix4f Q_trans;
  //     v_count = 0;
  //     for(size_t rQ=0; rQ<4; ++rQ){
  //       for(size_t cQ=rQ; cQ<4; ++cQ){
  //         float one_val = svd_thin_trans.matrixV()(v_count, 9);
  //         Q_trans(rQ, cQ) = one_val;
  //         Q_trans(cQ, rQ) = one_val;
  //         v_count++;
  //       }
  //     }
  //     // translate back to its position
  //     Eigen::Matrix4f Q = trans_m_inv.transpose() * Q_trans * trans_m_inv;
  //     std::cout << "--The estimated quadric is:" << std::endl << Q << std::endl;
  //     det = Q.determinant();
  //     if(det == 0)
  //       std::cout << "!!!Q is singular!!! " << std::endl;
  //     std::cout << "  The determinant of Q is " << det << std::endl; 
  //     // From Generic Quadric To Ellipsoid
  //     Q33 = Q.topLeftCorner(3,3);
  //     Q4 = Q.topRightCorner(3,1);
  //     trans_v = -1 * Q33.inverse() * Q4;
  //     std::cout << "  The translation parameters are:" << std::endl << trans_v << std::endl;
  //     float coef = -1 * det / Q33.determinant();
  //     std::cout << "  coefficient is: " << coef << std::endl;
  //     Eigen::EigenSolver<Eigen::Matrix3f> es(Q33);
  //     std::cout << "  eigenvalues:" << std::endl;
  //     std::cout << es.eigenvalues() << std::endl;
  //     std::cout << "  eigenvectors=" << std::endl;
  //     std::cout << es.eigenvectors() << std::endl;
  //     // axes
  //     // rotation matrix
  //     // translation matrix
  //   }
  // } // loop for # of detections
// }

} // namespace fusion