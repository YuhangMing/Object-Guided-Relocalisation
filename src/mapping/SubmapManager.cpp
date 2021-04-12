#include "mapping/SubmapManager.h"
#include "detection/map_cuboid.h"
#include "utils/settings.h"
#include <ctime>

namespace fusion
{

SubmapManager::SubmapManager() : bKFCreated(false) {
	// maxNumTriangle
	pMesher = new MeshEngine(20000000);
	pRayTracer = new RayTraceEngine(GlobalCfg.width, GlobalCfg.height, GlobalCfg.K);
}

SubmapManager::~SubmapManager()
{
	delete pMesher;
	delete pRayTracer;
	for(size_t i=0; i<vActiveSubmaps.size(); ++i)
	{
		delete vActiveSubmaps[i];
	}
	for(size_t i=0; i<vObjectMaps.size(); ++i)
	{
		delete vObjectMaps[i];
	}
}

void SubmapManager::Create(int submapIdx, bool bTrack, bool bRender)
{
	std::cout << "Create submap no. " << submapIdx << std::endl;

	auto pDenseMap = new MapStruct(GlobalCfg.K);
	pDenseMap->SetMeshEngine(pMesher);
	pDenseMap->SetTracer(pRayTracer);
	// hashTableSize, bucketSize, voxelBlockSize, voxelSize, truncationDist
	// pDenseMap->create(12000, 10000, 9000, 0.01, 0.1);
	// state.num_total_buckets_ = 50000;
    // state.num_total_hash_entries_ = 62500;
    // state.num_total_voxel_blocks_ = 50000;
    // state.num_max_rendering_blocks_ = 25000;
    // state.num_max_mesh_triangles_ = 5000000;
	pDenseMap->create(62500, 50000, 50000, 0.004, 0.012);	// delicate small maps
	// pDenseMap->create(62500, 50000, 50000, 0.008, 0.048);	// mid map
	// pDenseMap->create(62500, 50000, 50000, 0.01, 0.1);	// coarse large map
	pDenseMap->reset();
	pDenseMap->SetPose(Sophus::SE3d());

	vActiveSubmaps.push_back(pDenseMap);
	std::cout << " - Dense map created." << std::endl;

	// bHasNewSM = false;
	renderIdx = submapIdx;
	ref_frame_id = 0;

	//!! Remove vPoses after orthogonal issue in pose loading
	vSubmapPoses.push_back(Sophus::SE3d().matrix());

	// Semantics
	if(GlobalCfg.bSemantic)
	{
		auto pObjMap = new ObjectMap(submapIdx);
		vObjectMaps.push_back(pObjMap);
		std::cout << " - Object map created." << std::endl;
	}
}

void SubmapManager::AddKeyFrame(Eigen::Matrix4f kfPose){
	// store kf pose in object map
    vObjectMaps[renderIdx]->vKFs.push_back(kfPose);
}


std::vector<MapStruct *> SubmapManager::getDenseMaps()
{
	return vActiveSubmaps;
}

std::vector<Eigen::Matrix<float, 4, 4>> SubmapManager::GetKFPoses(){
	return vObjectMaps[renderIdx]->vKFs;
	
	// std::vector<Eigen::Matrix<float, 4, 4>> poses;
	// Eigen::Matrix4f Tw2rfinv = active_submaps[renderIdx]->poseGlobal.cast<float>().matrix().inverse();
    // // actives
    // for (size_t i=0; i<active_submaps.size(); ++i)
    // {
    // 	Eigen::Matrix4f Twm = active_submaps[i]->poseGlobal.cast<float>().matrix();
	// 	if(active_submaps[i]->vKFposes.size() > 0){
	// 		for(size_t j=0; j<active_submaps[i]->vKFposes.size(); ++j)
	// 		{
	// 			Eigen::Matrix4f Tmf = active_submaps[i]->vKFposes[j];
	// 			Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
	// 			poses.emplace_back(pose);
	// 		}
	// 	} else {
	// 		for(size_t j=0; j<active_submaps[i]->vKFs.size(); ++j){
	// 			Eigen::Matrix4f Tmf = active_submaps[i]->vKFs[j]->pose.cast<float>().matrix();
	// 			Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
	// 			poses.emplace_back(pose);	
	// 		}
	// 	}
    // }
    // // passives
    // for (size_t i=0; i<passive_submaps.size(); ++i)
    // {
    // 	Eigen::Matrix4f Twm = passive_submaps[i]->poseGlobal.cast<float>().matrix();
    // 	if(passive_submaps[i]->vKFposes.size()>0){
	// 		for(size_t j=0; j<passive_submaps[i]->vKFposes.size(); ++j)
	// 		{
	// 			Eigen::Matrix4f Tmf = passive_submaps[i]->vKFposes[j];
	// 			Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
	// 			poses.emplace_back(pose);
	// 		}
	// 	} else {
	// 		for(size_t j=0; j<passive_submaps[i]->vKFs.size(); ++j){
	// 			Eigen::Matrix4f Tmf = passive_submaps[i]->vKFs[j]->pose.cast<float>().matrix();
	// 			Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
	// 			poses.emplace_back(pose);	
	// 		}	
	// 	}
    // }
    // return poses;
}

std::vector<std::pair<int, std::vector<float>>> SubmapManager::GetObjects(bool bMain)
{
	// bMain stands for visualise main cuboids only or all cuboids
	std::vector<std::pair<int, std::vector<float>>> label_dim_pair;
	std::vector<std::shared_ptr<Object3d>>::iterator it;
	
	int mapIdx = 0; // change to a proper setup later

	for(it=vObjectMaps[renderIdx]->v_objects.begin(); it!=vObjectMaps[renderIdx]->v_objects.end(); ++it)
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

int SubmapManager::GetNumObjs() const
{
	return vObjectMaps[renderIdx]->v_objects.size();
}


void SubmapManager::writeMapToDisk()
{
	// dense maps with corresponding map poses
	std::string pose_file_name = GlobalCfg.map_file + "_poses.txt";
	std::vector<Eigen::Matrix4d> vPoseMaps;
	for(int i=0; i<vActiveSubmaps.size(); ++i)
	{
		std::string file_name = GlobalCfg.map_file + "_" + std::to_string(i) + ".data";
		auto pDenseMap = vActiveSubmaps[i];
		vPoseMaps.push_back(pDenseMap->GetPose().matrix());
		pDenseMap->writeToDisk(file_name);
	}
	writePosesToText(pose_file_name, vPoseMaps);

	// object maps with kfs that detect the objects
	for(int i=0; i<vObjectMaps.size(); ++i)
	{
		std::string file_name = GlobalCfg.map_file + "_" + std::to_string(i);
		vObjectMaps[i]->writeObjectsToDisk(file_name);
	}
}

void SubmapManager::writePosesToText(std::string file_name, 
									std::vector<Eigen::Matrix4d> vPoses)
{
	std::ofstream pose_file;
	pose_file.open(file_name, std::ios::out);
	if(pose_file.is_open())
	{
		for(auto Tmw : vPoses)
		{
			Eigen::Matrix4f fTmw = Tmw.cast<float>();
			Eigen::Matrix3f rot = fTmw.topLeftCorner<3,3>();
			Eigen::Vector3f trans = fTmw.topRightCorner<3,1>();
			Eigen::Quaternionf quat(rot);
			// std::cout << rot << std::endl << trans << std::endl;
			// std::cout << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << std::endl;
			pose_file << 0 << " " 
					<< trans(0) << " " << trans(1) << " " << trans(2) << " "
					<< quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
					<< "\n";			
		}
	}
	else
	{
		std::cout << "FAILED: cannot open the pose file." << std::endl;
	}
	pose_file.close();
}

void SubmapManager::readMapFromDisk()
{
	// dense maps with corresponding map poses
	std::string pose_file_name = GlobalCfg.map_file + "_poses.txt";
	// std::vector<Eigen::Matrix4d> vSubmapPoses;
	readPosesFromText(pose_file_name, vSubmapPoses);
	std::cout << "- " << vSubmapPoses.size() << " map poses loaded." << std::endl;
	// update the existing map
	std::cout << "Reading the dense maps:" << std::endl;
	for(int i=0; i<vActiveSubmaps.size(); ++i)
	{
		std::string file_name = GlobalCfg.map_file + "_" + std::to_string(i) + ".data";	
		std::cout << " " << file_name << std::endl;
		auto pDenseMap = vActiveSubmaps[i];
		pDenseMap->readFromDisk(file_name);
		pDenseMap->SetPose(Sophus::SE3d()); //vPoseMaps[i]
	}
	// create new dense maps, if necessary
	int num_active_maps = vActiveSubmaps.size();
	for(int i=num_active_maps; i<GlobalCfg.mapSize; ++i)
	{
		std::string file_name = GlobalCfg.map_file + "_" + std::to_string(i) + ".data";
		std::cout << " " << file_name << std::endl;
		auto pDenseMap = new MapStruct(GlobalCfg.K);
		pDenseMap->SetMeshEngine(pMesher);
		pDenseMap->SetTracer(pRayTracer);
		pDenseMap->readFromDisk(file_name);
		pDenseMap->SetPose(Sophus::SE3d()); //vPoseMaps[i]

		vActiveSubmaps.push_back(pDenseMap);

		renderIdx = 0; // ?? set to i or 0 here???
		ref_frame_id = 0;
	}

	if(GlobalCfg.bSemantic)
	{
		// object maps with kfs that detect the objects
		std::cout << "Reading the object maps:" << std::endl;
		// ONLY READ mapi_0 for now, update later
		std::string file_name = GlobalCfg.map_file + "_" + std::to_string(0);
		std::cout << " " << file_name << "_XXX.txt" << std::endl;
		if(vObjectMaps.size() > 0){
			vObjectMaps[0]->readObjectsFromDisk(file_name);
		}else{
			auto pObjectMap = new ObjectMap(0);
			pObjectMap->readObjectsFromDisk(file_name);
			vObjectMaps.push_back(pObjectMap);
		}
		// for(int i=0; i<vObjectMaps.size(); ++i)
		// {
		// 	std::string file_name = GlobalCfg.map_file + "_" + std::to_string(i);
		// 	std::cout << " " << file_name << std::endl;
		// 	vObjectMaps[i]->readObjectsFromDisk(file_name);
		// }
		// int num_object_maps = vObjectMaps.size();
		// for(int i=num_object_maps; i<GlobalCfg.mapSize; ++i)
		// {
		// 	std::string file_name = GlobalCfg.map_file + "_" + std::to_string(i);
		// 	std::cout << " " << file_name << std::endl;
		// 	auto pObjectMap = new ObjectMap(i);
		// 	pObjectMap->readObjectsFromDisk(file_name);
		// 	vObjectMaps.push_back(pObjectMap);
		// }
	}
}

void SubmapManager::readPosesFromText(std::string file_name, 
									  std::vector<Eigen::Matrix4d>& vPoses)
{
	vPoses.clear();
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
            std::cout << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w()
                      << ", " << t(0) << ", " << t(1) << ", " << t(2) << std::endl;
            Eigen::Matrix3d Rot = q.toRotationMatrix();
			// // Enforce orthogonal requirement !!!!!!! WRONG ROTATION RETURNED
			// Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rot, Eigen::ComputeThinU | Eigen::ComputeThinV);
			// // std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
			// // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
			// // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
			// Rot = svd.matrixU() * svd.matrixV();
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.topLeftCorner(3,3) = Rot;
            T.topRightCorner(3,1) = t;
            vPoses.push_back(T);
        }
    }
    else
    {
        std::cout << "FAILED: cannot find " << file_name << std::endl;
    }
    pose_file.close();
}

// test for map registration visualisation
void SubmapManager::CreateWithLoad(std::string map_path)
{
	// pose
	std::string pose_path = map_path + "_poses.txt";
	std::cout << "Reading map pose: " << pose_path << std::endl;
	std::ifstream pose_file(pose_path);
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
            std::cout << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w()
                      << ", " << t(0) << ", " << t(1) << ", " << t(2) << std::endl;
            Eigen::Matrix3d Rot = q.toRotationMatrix();
			// // Enforce orthogonal requirement !!!!!!! WRONG ROTATION RETURNED
			// Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rot, Eigen::ComputeThinU | Eigen::ComputeThinV);
			// // std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
			// // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
			// // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
			// Rot = svd.matrixU() * svd.matrixV();

			Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.topLeftCorner(3,3) = Rot;
            T.topRightCorner(3,1) = t;

            vSubmapPoses.push_back(T);
        }
    } else {
        std::cout << "FAILED: cannot find " << pose_path << std::endl;
    }
    pose_file.close();

	// dense map
	std::string file_name = map_path + "_4.data";
	std::cout << "Reading dense map: " << file_name << std::endl;
	auto pDenseMap = new MapStruct(GlobalCfg.K);
	pDenseMap->SetMeshEngine(pMesher);
	pDenseMap->SetTracer(pRayTracer);
	pDenseMap->readFromDisk(file_name);
	pDenseMap->SetPose(Sophus::SE3d()); //vPoseMaps[i]

	vActiveSubmaps.push_back(pDenseMap);

	renderIdx = 0; // ?? set to i or 0 here???
	ref_frame_id = 0;

}

void SubmapManager::ResetSubmaps(){
	// for(size_t i=0; i < active_submaps.size(); i++){
	// 	active_submaps[i]->reset_mapping();
	// }
	// for(size_t i=0; i < passive_submaps.size(); i++){
	// 	passive_submaps[i]->reset_mapping();
	// }

	// // submap storage
	// active_submaps.clear();
	// passive_submaps.clear();
	// activeTOpassiveIdx.clear();
}


/* Submapping disabled for now, need to be rewritten
void SubmapManager::Create(int submapIdx, RgbdImagePtr ref_img, bool bTrack, bool bRender)
{
	// std::cout << "Create submap no. " << submapIdx << std::endl;

	// auto ref_frame = ref_img->get_reference_frame();
	// // create new submap
	// auto submap = std::make_shared<DenseMapping>(GlobalCfg.K, GlobalCfg.width, GlobalCfg.height,
	// 											 submapIdx, bTrack, bRender);
	// submap->poseGlobal = active_submaps[renderIdx]->poseGlobal * ref_frame->pose;
	// // store new submap
	// active_submaps.push_back(submap);
	// // stop previous rendering submap from fusing depth info
	// active_submaps[renderIdx]->bRender = false;

	// // create new model frame for tracking and rendering
	// auto model_i = std::make_shared<DeviceImage>(ref_img->vKInv);
	// copyDeviceImage(ref_img, model_i);
	// auto model_f = model_i->get_reference_frame();	// new frame created when perform copy above, new pointer here
	// model_f->pose = Sophus::SE3d();	// every new submap starts its own reference coordinate system
	// odometry->vModelFrames.push_back(model_f);
	// odometry->vModelDeviceMapPyramid.push_back(model_i);

	// // some other parameters
	// bHasNewSM = true;
	// renderIdx = active_submaps.size()-1;
	// ref_frame_id = ref_frame->id;
}

void SubmapManager::readSMapFromDisk(std::string file_name)
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

} // namespace fusion