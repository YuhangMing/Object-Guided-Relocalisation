#include "map_manager.h"
#include "data_struct/map_cuboid.h"
#include <ctime>

namespace fusion
{

SubMapManager::SubMapManager() : bKFCreated(false) {
	ResetSubmaps();
}

void SubMapManager::Create(const fusion::IntrinsicMatrix base, 
						   int submapIdx, bool bTrack, bool bRender)
{
	std::cout << "Create submap no. " << submapIdx << std::endl;

	auto submap = std::make_shared<DenseMapping>(base, submapIdx, bTrack, bRender);
	submap->poseGlobal = Sophus::SE3d();	// set to identity
	active_submaps.push_back(submap);

	bHasNewSM = false;
	renderIdx = submapIdx;
	ref_frame_id = 0;
}

void SubMapManager::Create(const fusion::IntrinsicMatrix base, 
						   int submapIdx, RgbdImagePtr ref_img, bool bTrack, bool bRender)
{
	std::cout << "Create submap no. " << submapIdx << std::endl;

	auto ref_frame = ref_img->get_reference_frame();
	// create new submap
	auto submap = std::make_shared<DenseMapping>(base, submapIdx, bTrack, bRender);
	submap->poseGlobal = active_submaps[renderIdx]->poseGlobal * ref_frame->pose;
	// store new submap
	active_submaps.push_back(submap);
	// stop previous rendering submap from fusing depth info
	active_submaps[renderIdx]->bRender = false;

	// create new model frame for tracking and rendering
	auto model_i = std::make_shared<DeviceImage>(base, ref_img->NUM_PYRS);
	copyDeviceImage(ref_img, model_i);
	auto model_f = model_i->get_reference_frame();	// new frame created when perform copy above, new pointer here
	model_f->pose = Sophus::SE3d();	// every new submap starts its own reference coordinate system
	odometry->vModelFrames.push_back(model_f);
	odometry->vModelDeviceMapPyramid.push_back(model_i);

	// some other parameters
	bHasNewSM = true;
	renderIdx = active_submaps.size()-1;
	ref_frame_id = ref_frame->id;
}

void SubMapManager::ResetSubmaps(){
	for(size_t i=0; i < active_submaps.size(); i++){
		active_submaps[i]->reset_mapping();
	}
	for(size_t i=0; i < passive_submaps.size(); i++){
		passive_submaps[i]->reset_mapping();
	}

	// submap storage
	active_submaps.clear();
	passive_submaps.clear();
	activeTOpassiveIdx.clear();
}

/* Submapping disabled for now, need to be rewritten
float SubMapManager::CheckVisPercent(int submapIdx){
	return active_submaps[submapIdx]->CheckVisPercent();
}

void SubMapManager::CheckActive(){
	for(size_t i=0; i < activeTOpassiveIdx.size(); i++){
		// std::cout << "Removing submap " << activeTOpassiveIdx[i] << std::endl;
		auto tmp_map = active_submaps[activeTOpassiveIdx[i]];

		// // store map points and normal in manager
		// mPassiveMPs.insert({tmp_map->submapIdx, tmp_map->vMapPoints});
		// mPassiveNs.insert({tmp_map->submapIdx, tmp_map->vMapNormals});
		// // Store all passive submaps in RAM, save memories on GPU
		// tmp_map->DownloadToRAM();
		// tmp_map->Release();

		passive_submaps.push_back(tmp_map);
		active_submaps.erase(active_submaps.begin() + activeTOpassiveIdx[i]);

		// delete corresponding model frame
		auto tmp_image = odometry->vModelDeviceMapPyramid[activeTOpassiveIdx[i]];
		auto tmp_frame = odometry->vModelFrames[activeTOpassiveIdx[i]];

		odometry->vModelDeviceMapPyramid.erase(odometry->vModelDeviceMapPyramid.begin() + activeTOpassiveIdx[i]);
		odometry->vModelFrames.erase(odometry->vModelFrames.begin() + activeTOpassiveIdx[i]);
		renderIdx--;
	}
	activeTOpassiveIdx.clear();
}

void SubMapManager::CheckTrackAndRender(int cur_frame_id, int max_perct_idx){
	if(bHasNewSM)
	{
		if(cur_frame_id - ref_frame_id >= 17){
			active_submaps[renderIdx]->bTrack = true;
			bHasNewSM = false;
			std::cout << "Start tracking on the new submap (" 
					  << ref_frame_id << "-" << cur_frame_id
					  << ")" << std::endl;
		}
	}
	else
	{
		// currently disabled !!!!!!!!!!!!!!!!!!!!!!!!!!
		// if(max_perct_idx != renderIdx){
		// 	active_submaps[renderIdx]->bRender = false;
		//  active_submaps[renderIdx]->bTrack = false;
		// 	active_submaps[max_perct_idx]->bRender = true;
		// 	active_submaps[max_perct_idx]->bTrack = true;
		// 	renderIdx = max_perct_idx;
		// }
	}
}

void SubMapManager::AddKeyFrame(RgbdFramePtr currKF){
	// extract key points from current kf
	cv::Mat source_image = currKF->image;
    auto frame_pose = currKF->pose.cast<float>();

    // std::clock_t start = std::clock();
    // cv::Mat raw_descriptors;
    // std::vector<cv::KeyPoint> raw_keypoints;
    // extractor->extract_features_surf(
    //     source_image,
    //     raw_keypoints,
    //     raw_descriptors);
    // std::cout << "# of raw keypoints is " << raw_keypoints.size() << std::endl;

    // extractor->compute_3d_points(
    //     currKF->vmap,
    //     currKF->nmap,
    //     raw_keypoints,
    //     raw_descriptors,
    //     currKF->cv_key_points,
    //     currKF->descriptors,
    //     currKF->key_points,
    //     frame_pose);

	// copy a version to store
	auto kf = std::make_shared<RgbdFrame>();
    currKF->copyTo(kf);
    active_submaps[renderIdx]->vKFs.push_back(kf);
    // std::cout << "# of 3d keypoints is " << kf->key_points.size() << std::endl;

    // std::cout << "Detecting SURF keypoints takes "
    // 		  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
    //           << " seconds" << std::endl;
}

std::vector<Eigen::Matrix<float, 4, 4>> SubMapManager::GetKFPoses(){
	std::vector<Eigen::Matrix<float, 4, 4>> poses;
	Eigen::Matrix4f Tw2rfinv = active_submaps[renderIdx]->poseGlobal.cast<float>().matrix().inverse();
    
    // actives
    for (size_t i=0; i<active_submaps.size(); ++i)
    {
    	Eigen::Matrix4f Twm = active_submaps[i]->poseGlobal.cast<float>().matrix();
		if(active_submaps[i]->vKFposes.size() > 0){
			for(size_t j=0; j<active_submaps[i]->vKFposes.size(); ++j)
			{
				Eigen::Matrix4f Tmf = active_submaps[i]->vKFposes[j];
				Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
				poses.emplace_back(pose);
			}
		} else {
			for(size_t j=0; j<active_submaps[i]->vKFs.size(); ++j){
				Eigen::Matrix4f Tmf = active_submaps[i]->vKFs[j]->pose.cast<float>().matrix();
				Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
				poses.emplace_back(pose);	
			}
		}
    }

    // passives
    for (size_t i=0; i<passive_submaps.size(); ++i)
    {
    	Eigen::Matrix4f Twm = passive_submaps[i]->poseGlobal.cast<float>().matrix();
    	if(passive_submaps[i]->vKFposes.size()>0){
			for(size_t j=0; j<passive_submaps[i]->vKFposes.size(); ++j)
			{
				Eigen::Matrix4f Tmf = passive_submaps[i]->vKFposes[j];
				Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
				poses.emplace_back(pose);
			}
		} else {
			for(size_t j=0; j<passive_submaps[i]->vKFs.size(); ++j){
				Eigen::Matrix4f Tmf = passive_submaps[i]->vKFs[j]->pose.cast<float>().matrix();
				Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
				poses.emplace_back(pose);	
			}	
		}
    }

    return poses;
}

void SubMapManager::GetPoints(float *pt3d, size_t &count, size_t max_size){
	// count = 0;
	// Sophus::SE3f Tw2rfinv = active_submaps[renderIdx]->poseGlobal.cast<float>().inverse();
	
	// // actives
	// for(size_t i=0; i<active_submaps.size(); ++i)
	// {
	// 	auto sm = active_submaps[i];
	// 	Sophus::SE3f Twm = sm->poseGlobal.cast<float>();
	// 	for(size_t j=0; j<sm->vKFs.size(); ++j)
	// 	{
	// 		auto kf = sm->vKFs[j];
	// 		for(size_t k=0; k<kf->key_points.size(); ++k)
	// 		{
	// 			auto pt = kf->key_points[k];
	// 			Eigen::Vector3f mp_global_renderSM = Tw2rfinv * Twm * pt->pos;
	// 			pt3d[count * 3 + 0] = mp_global_renderSM(0);
	//             pt3d[count * 3 + 1] = mp_global_renderSM(1);
	//             pt3d[count * 3 + 2] = mp_global_renderSM(2);
	//             count++;
	// 		}  // pts
	// 	}  // kfs
    // }  // sms

    // // passives
    // for(size_t i=0; i<passive_submaps.size(); ++i)
	// {
	// 	auto sm = passive_submaps[i];
	// 	Sophus::SE3f Twm = sm->poseGlobal.cast<float>();
	// 	for(size_t j=0; j<sm->vKFs.size(); ++j)
	// 	{
	// 		auto kf = sm->vKFs[j];
	// 		for(size_t k=0; k<kf->key_points.size(); ++k)
	// 		{
	// 			auto pt = kf->key_points[k];
	// 			Eigen::Vector3f mp_global_renderSM = Tw2rfinv * Twm * pt->pos;
	// 			pt3d[count * 3 + 0] = mp_global_renderSM(0);
	//             pt3d[count * 3 + 1] = mp_global_renderSM(1);
	//             pt3d[count * 3 + 2] = mp_global_renderSM(2);
	//             count++;
	// 		}  // pts
	// 	}  // kfs
    // }  // sms

    // // std::cout << "NUM KEY POINTS: " << count << std::endl;
}

std::vector<std::pair<int, std::vector<float>>> SubMapManager::GetObjects(bool bMain)
{
	std::vector<std::pair<int, std::vector<float>>> label_dim_pair;

	std::vector<std::shared_ptr<Object3d>>::iterator it;

	for(it=active_submaps[renderIdx]->v_objects.begin(); it!=active_submaps[renderIdx]->v_objects.end(); ++it)
	{
		int label = (*it)->label;
		
		if(bMain){
			std::shared_ptr<Cuboid3d> cub = (*it)->v_all_cuboids[(*it)->primary_cuboid_idx];
			std::vector<float> corners{cub->cuboid_corner_pts.at<float>(0,0), cub->cuboid_corner_pts.at<float>(1,0), cub->cuboid_corner_pts.at<float>(2,0),
								   	   cub->cuboid_corner_pts.at<float>(0,1), cub->cuboid_corner_pts.at<float>(1,1), cub->cuboid_corner_pts.at<float>(2,1),
								       cub->cuboid_corner_pts.at<float>(0,2), cub->cuboid_corner_pts.at<float>(1,2), cub->cuboid_corner_pts.at<float>(2,2),
								       cub->cuboid_corner_pts.at<float>(0,3), cub->cuboid_corner_pts.at<float>(1,3), cub->cuboid_corner_pts.at<float>(2,3),
								       cub->cuboid_corner_pts.at<float>(0,4), cub->cuboid_corner_pts.at<float>(1,4), cub->cuboid_corner_pts.at<float>(2,4),
								       cub->cuboid_corner_pts.at<float>(0,5), cub->cuboid_corner_pts.at<float>(1,5), cub->cuboid_corner_pts.at<float>(2,5),
								       cub->cuboid_corner_pts.at<float>(0,6), cub->cuboid_corner_pts.at<float>(1,6), cub->cuboid_corner_pts.at<float>(2,6),
								       cub->cuboid_corner_pts.at<float>(0,7), cub->cuboid_corner_pts.at<float>(1,7), cub->cuboid_corner_pts.at<float>(2,7)};
			label_dim_pair.push_back(std::make_pair(label, corners));
		}
		else
		{
			for(size_t c=0; c<(*it)->v_all_cuboids.size(); ++c)
			{
				std::shared_ptr<Cuboid3d> cub = (*it)->v_all_cuboids[c];
				std::vector<float> corners{cub->cuboid_corner_pts.at<float>(0,0), cub->cuboid_corner_pts.at<float>(1,0), cub->cuboid_corner_pts.at<float>(2,0),
										cub->cuboid_corner_pts.at<float>(0,1), cub->cuboid_corner_pts.at<float>(1,1), cub->cuboid_corner_pts.at<float>(2,1),
										cub->cuboid_corner_pts.at<float>(0,2), cub->cuboid_corner_pts.at<float>(1,2), cub->cuboid_corner_pts.at<float>(2,2),
										cub->cuboid_corner_pts.at<float>(0,3), cub->cuboid_corner_pts.at<float>(1,3), cub->cuboid_corner_pts.at<float>(2,3),
										cub->cuboid_corner_pts.at<float>(0,4), cub->cuboid_corner_pts.at<float>(1,4), cub->cuboid_corner_pts.at<float>(2,4),
										cub->cuboid_corner_pts.at<float>(0,5), cub->cuboid_corner_pts.at<float>(1,5), cub->cuboid_corner_pts.at<float>(2,5),
										cub->cuboid_corner_pts.at<float>(0,6), cub->cuboid_corner_pts.at<float>(1,6), cub->cuboid_corner_pts.at<float>(2,6),
										cub->cuboid_corner_pts.at<float>(0,7), cub->cuboid_corner_pts.at<float>(1,7), cub->cuboid_corner_pts.at<float>(2,7)};
				label_dim_pair.push_back(std::make_pair(label, corners));
			}
		}
	}

	// return dimensions;
	return label_dim_pair;
}
std::vector<std::pair<int, std::vector<float>>> SubMapManager::GetObjectCuboids()
{
	std::vector<std::pair<int, std::vector<float>>> label_dim_pair;

	// // std::vector<std::shared_ptr<Cuboid3d>>::iterator it;
	// // for(it=active_submaps[renderIdx]->object_cuboids.begin(); it!=active_submaps[renderIdx]->object_cuboids.end(); ++it){
	// // 	std::vector<float> corners{(*it)->cuboid_corner_pts.at<float>(0,0), (*it)->cuboid_corner_pts.at<float>(1,0), (*it)->cuboid_corner_pts.at<float>(2,0),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,1), (*it)->cuboid_corner_pts.at<float>(1,1), (*it)->cuboid_corner_pts.at<float>(2,1),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,2), (*it)->cuboid_corner_pts.at<float>(1,2), (*it)->cuboid_corner_pts.at<float>(2,2),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,3), (*it)->cuboid_corner_pts.at<float>(1,3), (*it)->cuboid_corner_pts.at<float>(2,3),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,4), (*it)->cuboid_corner_pts.at<float>(1,4), (*it)->cuboid_corner_pts.at<float>(2,4),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,5), (*it)->cuboid_corner_pts.at<float>(1,5), (*it)->cuboid_corner_pts.at<float>(2,5),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,6), (*it)->cuboid_corner_pts.at<float>(1,6), (*it)->cuboid_corner_pts.at<float>(2,6),
	// // 							   (*it)->cuboid_corner_pts.at<float>(0,7), (*it)->cuboid_corner_pts.at<float>(1,7), (*it)->cuboid_corner_pts.at<float>(2,7)};
		
	// // 	label_dim_pair.push_back(std::make_pair((*it)->label, corners));
	// // }

	// std::map<int, std::vector<std::shared_ptr<Cuboid3d>>>::iterator it;
	// for(it=active_submaps[renderIdx]->cuboid_dictionary.begin(); it!=active_submaps[renderIdx]->cuboid_dictionary.end(); ++it)
	// {
	// 	for(size_t i=0; i< it->second.size(); ++i)
	// 	{
	// 		// if(it->second[i]->observation < 9)
	// 		// 	continue;
			
	// 		std::vector<float> corners{it->second[i]->cuboid_corner_pts.at<float>(0,0), it->second[i]->cuboid_corner_pts.at<float>(1,0), it->second[i]->cuboid_corner_pts.at<float>(2,0),
	// 							   	   it->second[i]->cuboid_corner_pts.at<float>(0,1), it->second[i]->cuboid_corner_pts.at<float>(1,1), it->second[i]->cuboid_corner_pts.at<float>(2,1),
	// 							       it->second[i]->cuboid_corner_pts.at<float>(0,2), it->second[i]->cuboid_corner_pts.at<float>(1,2), it->second[i]->cuboid_corner_pts.at<float>(2,2),
	// 							       it->second[i]->cuboid_corner_pts.at<float>(0,3), it->second[i]->cuboid_corner_pts.at<float>(1,3), it->second[i]->cuboid_corner_pts.at<float>(2,3),
	// 							       it->second[i]->cuboid_corner_pts.at<float>(0,4), it->second[i]->cuboid_corner_pts.at<float>(1,4), it->second[i]->cuboid_corner_pts.at<float>(2,4),
	// 							       it->second[i]->cuboid_corner_pts.at<float>(0,5), it->second[i]->cuboid_corner_pts.at<float>(1,5), it->second[i]->cuboid_corner_pts.at<float>(2,5),
	// 							       it->second[i]->cuboid_corner_pts.at<float>(0,6), it->second[i]->cuboid_corner_pts.at<float>(1,6), it->second[i]->cuboid_corner_pts.at<float>(2,6),
	// 							       it->second[i]->cuboid_corner_pts.at<float>(0,7), it->second[i]->cuboid_corner_pts.at<float>(1,7), it->second[i]->cuboid_corner_pts.at<float>(2,7)};
		
	// 		label_dim_pair.push_back(std::make_pair(it->first, corners));
	// 	}
		
	// }

	// return dimensions;
	return label_dim_pair;
}

std::vector<float> SubMapManager::GetObjectCentroidAxes(int idx_obj){
	std::vector<float> vAxes;
	// for(size_t c=0; c<4; ++c){
	// 	for(size_t r=0; r<3; ++r){
	// 		vAxes.push_back(active_submaps[renderIdx]->object_cuboids[idx_obj]->axes.at<float>(r,c));
	// 	}
	// }
	// std::cout << vAxes.size() << std::endl;
	return vAxes;
}

int SubMapManager::GetObjectPts(float *points, size_t &count, int idx_obj){
	// int num = active_submaps[renderIdx]->object_cuboids[idx_obj]->v_data_pts.size(),
	// 	max = 1000000/3;
	// count = num < max ? num : max;
    // // std::cout << count << " valid pts found" << std::endl;
    
    // for(size_t i=0; i<count; ++i)
    // {
	// 	// int idx = rand() % num;
    //     for(size_t j=0; j<3; ++j)
    //     {
    //         points[i*3+j] = active_submaps[renderIdx]->object_cuboids[idx_obj]->v_data_pts[i](j);
	// 	}
    // }
    
    // return active_submaps[renderIdx]->object_cuboids[idx_obj]->label;

	return 0;
}

int SubMapManager::GetNumObjs() const
{
	return active_submaps[renderIdx]->v_objects.size();
}

int SubMapManager::GetObjectPointCloud(float *points, size_t &count, int idx_obj)
{
	// count = active_submaps[renderIdx]->better_cuboids[idx_obj]->data_pts.rows;
	// for(size_t i=0; i<count; ++i)
	// {
	// 	for(size_t j=0; i<3; ++j)
	// 	{
	// 		points[i*3+j] = active_submaps[renderIdx]->better_cuboids[idx_obj]->data_pts.at<float>(i,j);
	// 	}
	// }

	return active_submaps[renderIdx]->v_objects[idx_obj]->label;
}

std::vector<float> SubMapManager::GetPlaneNormals()
{
	std::vector<float> plane_norms;
	if(active_submaps[renderIdx]->plane_normals.size() > 0){
		for(size_t i=0; i<3; ++i){
			for(size_t j=0; j<3; ++j){
				plane_norms.push_back(active_submaps[renderIdx]->plane_normals[i](j));
			}
		}
	}
	return plane_norms;
}

void SubMapManager::SetTracker(std::shared_ptr<DenseOdometry> pOdometry){
	odometry = pOdometry;
}
*/

void SubMapManager::readSMapFromDisk(std::string file_name)
{
	// // semantic map
	// for(size_t idx=1; idx<7; ++idx){
	// 	std::vector<std::shared_ptr<Cuboid3d>> one_map;
	// 	std::ifstream semanticfile(file_name+std::to_string(idx)+".data-semantic.txt", std::ios::in);
	// 	int numObj;
	// 	if (semanticfile.is_open())
	// 	{
	// 		std::string sNumObj;
	// 		std::getline(semanticfile, sNumObj);
	// 		numObj = std::stoi(sNumObj);
	// 		std::cout << sNumObj + " objects in the map." << std::endl;
	// 	}
	// 	semanticfile.close();
	// 	for(size_t i=0; i<numObj; ++i)
	// 	{
	// 		std::shared_ptr<Cuboid3d> new_cuboid(new Cuboid3d(file_name+std::to_string(idx)+".data", i*6+1));
	// 		one_map.push_back(std::move(new_cuboid));
	// 	}

	// 	// store the map
	// 	vSMaps.push_back(one_map);
	// }
}

} // namespace fusion