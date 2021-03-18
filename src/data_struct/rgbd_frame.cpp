#include "data_struct/rgbd_frame.h"
#include <ctime>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>

namespace fusion
{

RgbdFrame::RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id, const double ts)
    : id(id), timeStamp(ts)
{
    this->image = image.clone();
	// img_original = image.clone();
    this->depth = depth.clone();
    row_frame = image.rows;
    col_frame = image.cols;
	cent_matrix = cv::Mat::zeros(cv::Size(3, 30), CV_32FC1);	// 6*5 rows and 3 cols, assuming 5 instances per class maximum
    numDetection = 0;
}

RgbdFrame::RgbdFrame()
{
	id = -1;
	timeStamp = -1.;
}

void RgbdFrame::copyTo(RgbdFramePtr dst){
	if(dst==NULL)
		return;

	dst->cv_key_points = cv_key_points;
	dst->key_points = key_points;
	dst->neighbours = neighbours;
	descriptors.copyTo(dst->descriptors);

	dst->id = id;
	dst->timeStamp = timeStamp;
	dst->pose = pose;

	image.copyTo(dst->image);
	depth.copyTo(dst->depth);
	vmap.copyTo(dst->vmap);
	nmap.copyTo(dst->nmap);

	dst->row_frame = row_frame;
	dst->col_frame = col_frame;
	
	dst->numDetection = numDetection;
	dst->vMasks = vMasks;
	dst->vLabels = vLabels;
	dst->vScores = vScores;
	dst->vCoords = vCoords;
	dst->vObjects = vObjects;
	cent_matrix.copyTo(dst->cent_matrix);
	mask.copyTo(dst->mask);
	nocs_map.copyTo(dst->nocs_map);
	
	mEdge.copyTo(dst->mEdge);
	dst->nConComps = nConComps;
	mLabeled.copyTo(dst->mLabeled);
	mStats.copyTo(dst->mStats);
	mCentroids.copyTo(dst->mCentroids);

	dst->plane_normals = plane_normals;
}

void RgbdFrame::ExtractSemantics(semantic::Detector* detector, bool bBbox, bool bContour, bool bText)
{
	ExtractObjects(detector, bBbox, bContour, bText);
	// std::cout << "[ " << numDetection << " objects detected ]" << std::endl;

	// // For the purpose of better poes-based reloc, no need if we only use centroid
	// ExtractPlanes();
	// std::cout << "[ " << numDetection << " objects and " 
	//           << plane_normals.size() << " plane normals found in current frame. ]" << std::endl;
	// // align objects with plane normals
	// // find the desired axis to be aligned with
	// std::vector<std::pair<int, int>> vpBestAxis; // first: index, second: sign
	// std::vector<int> vCount;
	// for(size_t i=0; i<vCuboids.size(); ++i){
	// 	int tmpIdx, tmpSign;
	// 	bool bAddNewAxis = true;
	// 	tmpIdx = vCuboids[i]->find_axis_correspondence(plane_normals, tmpSign);
	// 	for(size_t j=0; j<vpBestAxis.size(); ++j)
	// 	{
	// 		std::pair<int, int> axis = vpBestAxis[j];
	// 		if(tmpIdx == axis.first && tmpSign == axis.second){
	// 			vCount[j]++;
	// 			bAddNewAxis = false;
	// 		}
	// 	}
	// 	if(bAddNewAxis){
	// 		vpBestAxis.push_back(std::make_pair(tmpIdx, tmpSign));
	// 		vCount.push_back(1);
	// 	}
	// }
	// // determine the most desired target axis
	// int bestAxisIdx = std::max_element(vCount.begin(),vCount.end()) - vCount.begin();
	// Eigen::Vector3f desired_axis = vpBestAxis[bestAxisIdx].second * plane_normals[vpBestAxis[bestAxisIdx].first];
	// for(size_t i=0; i<vCuboids.size(); ++i){
	// 	// vCuboids[i]->align_with_palne_normal(desired_axis);
	// 	vCuboids[i]->find_bounding_cuboid();
	// }
}

void RgbdFrame::ExtractPlanes()
{
	std::clock_t start = std::clock();
	// std::cout << mask.type() << " - " << nmap.type() << std::endl;
	// mask CV_32SC1; nmap CV_32SC4
	// initialize means
	Eigen::Vector3f meanX(1, 0, 0);
	Eigen::Vector3f meanY(0, 1, 0);
	Eigen::Vector3f meanZ(0, 0, 1);
	std::vector<Eigen::Vector3f> cluster_means{meanX, meanY, meanZ};
	std::vector<float> cluster_means_shift{100, 1000, 1000};
	int iter_count = 0;
	while(true){
		iter_count++;
		std::vector<Eigen::Vector3f> cluster_means_updated{Eigen::Vector3f::Zero(),
														Eigen::Vector3f::Zero(),
														Eigen::Vector3f::Zero()};
		std::vector<int> cluster_size{0, 0, 0};
		// loop through all pts to update means
		for(size_t r=0; r<mask.rows; ++r)
		{
			for(size_t c=0; c<mask.cols; ++c)
			{
				if(mask.at<int>(r,c) != 0 || nmap.at<cv::Vec4f>(r,c) != nmap.at<cv::Vec4f>(r,c)){
					continue;
				}
				Eigen::Vector3f one_normal(nmap.at<cv::Vec4f>(r,c)[0]/nmap.at<cv::Vec4f>(r,c)[3],
										nmap.at<cv::Vec4f>(r,c)[1]/nmap.at<cv::Vec4f>(r,c)[3],
										nmap.at<cv::Vec4f>(r,c)[2]/nmap.at<cv::Vec4f>(r,c)[3]);
				float dist0 = (one_normal - cluster_means[0]).norm(),
					dist1 = (one_normal - cluster_means[1]).norm(),
					dist2 = (one_normal - cluster_means[2]).norm();
				if(dist0 <= dist1 && dist0 <= dist2){
					cluster_means_updated[0] += one_normal;
					cluster_size[0]++;
				}
				if(dist1 <= dist0 && dist1 <= dist2){
					cluster_means_updated[1] += one_normal;
					cluster_size[1]++;
				}
				if(dist2 <= dist0 && dist2 <= dist1){
					cluster_means_updated[2] += one_normal;
					cluster_size[2]++;
				}
			}
		}
		// update means and check convergence
		for(size_t i=0; i<3; ++i)
		{
			cluster_means_updated[i] /= cluster_size[i];
			cluster_means_shift[i] = (cluster_means_updated[i] - cluster_means[i]).norm();
		}
		if(cluster_means_shift[0]<0.0001 && cluster_means_shift[1]<0.0001 && cluster_means_shift[2]<0.0001){
			break;
		} else {
			cluster_means = cluster_means_updated;
		}
	}

	// store the normals as normalized means
	Eigen::Matrix4f cam_pose_f = pose.matrix().cast<float>();
	for(size_t i=0; i<3; ++i){
		// Eigen::Vector3f tmp = cam_pose_f.topLeftCorner(3,3)*cluster_means[i]+cam_pose_f.topRightCorner(3,1);
		Eigen::Vector3f tmp = cluster_means[i]; 
		plane_normals.push_back(tmp/tmp.norm());
	}
	// plane_normals = cluster_means;

	std::cout << "#### K-Means takes "
			  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC
			  << " seconds and converges in " << iter_count << " iterations with last mean-shift: " 
			  << cluster_means_shift[0] << ", " 
			  << cluster_means_shift[1] << ", " 
			  << cluster_means_shift[2] << std::endl;
	
}

void RgbdFrame::ExtractObjects(semantic::Detector* detector, bool bBbox, bool bContour, bool bText)
{
	// detection
	std::clock_t start = std::clock();
	switch(detector->detector_name){
		case Detector_MaskRCNN:
			detector->performDetection(image);
			break;
		case Detector_NOCS:
			detector->performDetection(image, depth);
			break;
		default:
			std::cout << "[ERROR] Detector loaded is NOT supported." << std::endl;
	}
	
	std::cout << "**** Detection in total takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << "seconds." << std::endl;
    start = std::clock();

	// get detected information
	numDetection = detector->numDetection;
	std::cout << "# of detected object passed to c++: " << numDetection << std::endl;

    vMasks.clear();
	vLabels.clear();
	vScores.clear();
	vCoords.clear();
	vObjects.clear();
	int discard_num = 0;
	for(size_t i=0; i<numDetection; ++i)
	{
		//- convert values from array to vector/Mat
		int tmpLabel = int(detector->pLabels[i]);
		float tmpScore = float(detector->pScores[i]); 
		// Only keep detection with enough confidence
		if(tmpScore < 0.7){
			discard_num ++;
			continue;
		}
		cv::Mat tmpMask = Array2Mat(&detector->pMasks[i*row_frame*col_frame]);
		// std::cout << "label=" << tmpLabel << ", score=" << tmpScore << std::endl;
		std::vector<cv::Mat> vTmpCoord;
		for(size_t d=0; d<3; ++d){
			cv::Mat tmpCoord = Array2Mat(&detector->pCoord[(i*3+d)*row_frame*col_frame]);
			vTmpCoord.push_back(tmpCoord.clone());
		}
		cv::Mat Coord;
		cv::merge(vTmpCoord, Coord);
		
		//- compute the pose of the detected object
		Eigen::Matrix4f tmpRt = Eigen::Matrix4f::Zero();
		std::vector<float> tmpDims;
		float tmpScale = 0;
		Eigen::Matrix3d tmpS_t = Eigen::Matrix3d::Identity();
		// std::cout << " - Obj with label " << tmpLabel << " and confidence " << tmpScore << std::endl;
		std::vector<Eigen::Vector3f> pt_cloud = FitPose(Coord, tmpMask, tmpRt, tmpDims, tmpScale, tmpS_t);
		if(tmpRt.isZero()){
			// std::cout << " !!! Zero Matrix returned, discard current detection." << std::endl;
			discard_num ++;
			continue;
		}

		//- inialize cuboid in the frame
		// double check whether one object was detected as 2 different ones
		std::shared_ptr<Cuboid3d> one_cuboid(new Cuboid3d(tmpLabel, tmpScore, tmpRt.cast<double>(), tmpDims, tmpScale, tmpS_t));
		one_cuboid->add_points(pt_cloud);
		std::shared_ptr<Object3d> one_object(new Object3d(one_cuboid));

		double new_dimx = double(one_cuboid->dims[0]*one_cuboid->scale/2),
               new_dimy = double(one_cuboid->dims[1]*one_cuboid->scale/2),
               new_dimz = double(one_cuboid->dims[2]*one_cuboid->scale/2);
		// double new_thre_max = std::max(new_dimx, new_dimz);
		double new_thre_max = sqrt(new_dimx*new_dimx + new_dimz*new_dimz);
		bool bOverlap = false;
		int update_idx = -1;
		// std::cout << one_object->label << ": \n";
		for(size_t j=0; j<vObjects.size(); ++j)
		{
			double pre_dimx = double(vObjects[j]->v_all_cuboids[0]->dims[0]*vObjects[j]->v_all_cuboids[0]->scale/2),
        	       pre_dimy = double(vObjects[j]->v_all_cuboids[0]->dims[1]*vObjects[j]->v_all_cuboids[0]->scale/2),
            	   pre_dimz = double(vObjects[j]->v_all_cuboids[0]->dims[2]*vObjects[j]->v_all_cuboids[0]->scale/2);
			// double pre_thre_max = std::max(pre_dimx, pre_dimz);
			double pre_thre_max = sqrt(pre_dimx*pre_dimx + pre_dimz*pre_dimz);
			double cent_thre_max = std::max(pre_thre_max, new_thre_max);
			if(vObjects[j]->label != one_object->label)
				cent_thre_max /= 2;
			Eigen::Vector3d diff = one_object->pos - vObjects[j]->pos;
			double distance_xz = sqrt(diff(0)*diff(0) + diff(2)*diff(2));
			// bool pre_in_new = std::abs(diff(0))<pre_dimx && std::abs(diff(1))<pre_dimy && std::abs(diff(2))<pre_dimz;
			// bool new_in_pre = std::abs(diff(0))<new_dimx && std::abs(diff(1))<new_dimy && std::abs(diff(2))<new_dimz;
			// std::cout << sqrt(diff(0)*diff(0) + diff(2)*diff(2)) << " - "
			// 		  << sqrt(diff(1)*diff(1) + diff(2)*diff(2)) << " - "
			// 		  << sqrt(diff(0)*diff(0) + diff(1)*diff(1)) << " - " 
			// 		  << cent_thre_max << "\n";
			// std::cout << "diff: " << diff(0) << " - " << diff(1) << " - " << diff(2) << "\n"
			// 		  << "new: " << new_dimx << " - " << new_dimy << " - " << new_dimz << "\n"
			// 		  << "pre: " << pre_dimx << " - " << pre_dimy << " - " << pre_dimz << "\n"
			// 		  << one_cuboid->confidence << " - " << vObjects[j]->v_all_cuboids[0]->confidence 
			// 		  << std::endl;
			// std::cout << (distance_xz<cent_thre_max) << " - "
			// 		  << pre_in_new << " - " << new_in_pre << std::endl;
			if( distance_xz<cent_thre_max )
			{
				bOverlap = true;
				if(one_cuboid->confidence > vObjects[j]->v_all_cuboids[0]->confidence)
					update_idx = j;
				// std::cout << "dist = " << distance_xz << " - " << cent_thre_max << std::endl
				// 		  << one_cuboid->confidence << " - " << vObjects[j]->v_all_cuboids[0]->confidence 
				// 		  << std::endl;
				// std::cout << "Same object " << vObjects[j]->label << " was detected as two different ones." << std::endl;
			}
		}
		// std::cout << "Trying initialize new object" << std::endl;
		if(!bOverlap){
			// store the detection
			vLabels.push_back(tmpLabel);
			vScores.push_back(tmpScore);
			vMasks.push_back(tmpMask.clone());
			vCoords.push_back(Coord.clone());
			vObjects.push_back(std::move(one_object));
		} else {
			discard_num ++;
			if(update_idx >= 0){
				vLabels[update_idx] = tmpLabel;
				vScores[update_idx] = tmpScore;
				vMasks[update_idx] = tmpMask.clone();
				vCoords[update_idx] = Coord.clone();
				vObjects[update_idx] = std::move(one_object);
			}
		}
		// std::cout<< "Initialized object " << vObjects.back()->label << std::endl;
		
		// //- display results
		// // combine masks into a single mask & draw detections
		// // new mask will have value as corresponding obj label
		// if(i-discard_num == 0){
		// 	mask = tmpMask*int(tmpLabel);
		// 	nocs_map = Coord.clone();
		// } else {
		// 	mask += tmpMask*int(tmpLabel);
		// 	nocs_map += Coord;
		// }
		// // draw bboxes
    	// cv::Scalar objColor = CalculateColor(tmpLabel);
    	// if (bBbox){
		// 	cv::Point2f top_left(detector->pBoxes[i*4], detector->pBoxes[i*4+1]);
		// 	cv::Point2f bottom_right(detector->pBoxes[i*4+2], detector->pBoxes[i*4+3]);
		// 	// cv::Point2f top_left(vBBoxes[(i-tmp_counter)][0], vBBoxes[(i-tmp_counter)][1]);
		// 	// cv::Point2f bottom_right(vBBoxes[(i-tmp_counter)][2], vBBoxes[(i-tmp_counter)][3]);
		// 	cv::rectangle(image, top_left, bottom_right, objColor);
		// }
		// // draw mask countors
		// if (bContour) {
		// 	// cv::Mat mMask = vMasks[i];
		// 	cv::Mat mMask;
		// 	tmpMask.convertTo(mMask, CV_8UC1);
		// 	std::vector<std::vector<cv::Point> > contours;
		// 	std::vector<cv::Vec4i> hierarchy;
		// 	cv::findContours(mMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		// 	cv::drawContours(image, contours, -1, objColor, 4);
		// }
		// // display text
		// if (bText){
		// 	std::string label_text = detector->CATEGORIES[vLabels[i]];
		// 	std::string score_text = std::to_string(tmpScore);
		// 	cv::Point2f top_left(detector->pBoxes[i*4], detector->pBoxes[i*4+1]);
		// 	cv::putText(image, label_text, top_left, cv::FONT_HERSHEY_SIMPLEX, 1.0, objColor);
		// }
	} // i
	numDetection -= discard_num;

	// release array memory in the detector class
	detector->releaseMemory();

	// store the centroids in the map, (row id % 6) = (object class id - 1)
	for(size_t i=0; i<numDetection; ++i)
	{
		int row = vObjects[i]->label - 1;
		while(cent_matrix.at<float>(row, 0)!=0 || 
			  cent_matrix.at<float>(row, 1)!=0 || 
			  cent_matrix.at<float>(row, 2)!=0)
		{ row += 6; }
		if(row >= 30)
			continue;
		Eigen::Vector3d pos = vObjects[i]->pos;
		// std::cout << "Stored in row " << row << ":" 
		// 		  << pos(0) << ", "  << pos(1) << ", "  << pos(2) << std::endl;
		cent_matrix.at<float>(row, 0) = pos(0);
		cent_matrix.at<float>(row, 1) = pos(1);
		cent_matrix.at<float>(row, 2) = pos(2);
	}
	// std::cout << cent_matrix << std::endl;

	// display detected result, move back to the loop once the above ensurance removed
	for(size_t i=0; i<numDetection; ++i){
		cv::Mat tmpMask = vMasks[i];
		cv::Mat tmpCoord = vCoords[i];
		// combine masks into a single mask & draw detections
		// new mask will have value as corresponding obj label
		if(i == 0){
			mask = tmpMask*int(vLabels[i]);
			nocs_map = tmpCoord.clone();
		} else {
			mask += tmpMask*int(vLabels[i]);
			nocs_map += tmpCoord;
		}
		// draw bboxes
    	cv::Scalar objColor = CalculateColor(vLabels[i]);
    	if (bBbox){
			std::cout << "Size of pBoxes and vBoxes doesn't match, check before display." << std::endl;
			// cv::Point2f top_left(detector->pBoxes[i*4], detector->pBoxes[i*4+1]);
			// cv::Point2f bottom_right(detector->pBoxes[i*4+2], detector->pBoxes[i*4+3]);
			// // cv::Point2f top_left(vBBoxes[(i-tmp_counter)][0], vBBoxes[(i-tmp_counter)][1]);
			// // cv::Point2f bottom_right(vBBoxes[(i-tmp_counter)][2], vBBoxes[(i-tmp_counter)][3]);
			// cv::rectangle(image, top_left, bottom_right, objColor);
		}
		// draw mask countors
		if (bContour) {
			// cv::Mat mMask = vMasks[i];
			cv::Mat mMask;
			tmpMask.convertTo(mMask, CV_8UC1);
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(mMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
			cv::drawContours(image, contours, -1, objColor, 4);
		}
		// display text
		if (bText){
			std::cout << "Size of pBoxes and vBoxes doesn't match, check before display." << std::endl;
			// std::string label_text = detector->CATEGORIES[vLabels[i]];
			// std::string score_text = std::to_string(vScores[i]);
			// cv::Point2f top_left(detector->pBoxes[i*4], detector->pBoxes[i*4+1]);
			// cv::putText(image, label_text, top_left, cv::FONT_HERSHEY_SIMPLEX, 1.0, objColor);
		}
	}


	std::cout << "# of detected object stored in frm: " << numDetection << std::endl;

	// Write CV::Mat to file, to check values
	// cv::FileStorage file("/home/yohann/some_name.xml", cv::FileStorage::WRITE);
		// // Write to file!
		// file << "matName" << vCoords[0];
		// std::ofstream fout("/home/yohann/some_name.txt");
		// if(!fout)
		// {
		//     std::cout<<"File Not Opened"<<std::endl;
		// }
		// for(int i=0; i<vCoords[0].rows; i++)
		// {
		//     for(int j=0; j<vCoords[0].cols; j++)
		//     {
		//         fout << vCoords[0].at<float>(i,j) << ",";
		//     }
		//     fout << std::endl;
		// }
    // fout.close();

	std::cout << "**** Estimate object pose takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << "s." << std::endl;
}

std::vector<Eigen::Vector3f> RgbdFrame::FitPose(cv::Mat coord, cv::Mat mask, 
												Eigen::Matrix4f& Twn, 
												std::vector<float>& dims, 
												float& scale,
												Eigen::Matrix3d& Sigma_t)
{
	// loop through every points in the matrix
	Eigen::Matrix4f Twf = pose.cast<float>().matrix();	// identity matrix when tracking lost
	float max_r=0, max_g=0, max_b=0;
	std::vector<Eigen::Vector3f> depth_pts, coord_pts;
	std::vector<Eigen::Vector3f> pt_cloud_wcs;
	// covariance
	int num_valid = 0;
	Eigen::Matrix3d avg_Sigma_bp = Eigen::Matrix3d::Zero();
	Eigen::Matrix3d Sigma_NOCS = Eigen::Matrix3d::Identity();
	for(size_t r=0; r<coord.rows; ++r)
	{
		for(size_t c=0; c<coord.cols; ++c)
		{
			if(mask.at<int>(r, c) < 1)
				continue;

			// update scale values
			cv::Vec3f coord_pt, coord_abs;
			// float [3];
			for(size_t d=0; d<3; ++d){
				coord_pt[d] = coord.at<cv::Vec3f>(r, c)[d] - 0.5;
				coord_abs[d] = std::abs(coord_pt[d]);
			}
			max_r = coord_abs[0] > max_r ? coord_abs[0] : max_r;
			max_g = coord_abs[1] > max_g ? coord_abs[1] : max_g;
			max_b = coord_abs[2] > max_b ? coord_abs[2] : max_b;

			// find valid 3d map points (backproject) using RAW DEPTH MAP
			if(depth.at<float>(r, c) > 0){
				// fx 580, fy 580, cx 319.5, cy 239.5; depth in meters
				float x3d, y3d, z3d;
				z3d = depth.at<float>(r, c);
				x3d = (float(c) - 319.5) * z3d / 580.;
				y3d = (float(r) - 239.5) * z3d / 580.;
				Eigen::Vector3f dep_pt, coo_pt;
				dep_pt << x3d, y3d, z3d;
				coo_pt << coord_pt[0], coord_pt[1], coord_pt[2];
				depth_pts.push_back(dep_pt);
				coord_pts.push_back(coo_pt);
				Eigen::Vector3f ptc = Twf.topLeftCorner(3,3) * dep_pt + Twf.topRightCorner(3,1);
				pt_cloud_wcs.push_back(ptc);

				// compute the covariance for this back-projected point
				Eigen::Matrix3d Sigma_s = Eigen::Matrix3d::Identity();
				Sigma_s(0, 0) = 1./12.;
				Sigma_s(1, 1) = 1./12.;
				Sigma_s(2, 2) = 1.425 * z3d * z3d;
				// std::cout << "Sigma_s:" << std::endl
				// 		  << Sigma_s << std::endl;

				// J = z/fx, 0,    (u-cx)/fx
				//     0,    z/fy, (v-cy)/fy
				//     0,    0,    1
				Eigen::Matrix3d J_proj = Eigen::Matrix3d::Identity();
				J_proj(0, 0) = z3d / 580.;
				J_proj(0, 2) = (float(c) - 319.5) / 580.;
				J_proj(1, 1) = z3d / 580.;
				J_proj(1, 2) = (float(r) - 239.5) / 580.;
				// std::cout << "J_proj:" << std::endl
				// 		  << J_proj << std::endl;
				
				Sigma_s = J_proj * Sigma_s * J_proj.transpose();
				// std::cout << "Sigma_s:" << std::endl
				// 		  << Sigma_s << std::endl;
				avg_Sigma_bp += Sigma_s;
				num_valid++;
			} 
		}
	}

	// Use the average as the final uncertainty
	avg_Sigma_bp /= num_valid;
	// std::cout << "avg_Sigma_bp:" << std::endl
	// 		  << avg_Sigma_bp << std::endl;

	// Perform similarity transform estimation
	Eigen::Matrix4f Tfn = RgbdFrame::EstimateSimilarityTransform(coord_pts, depth_pts, scale);
	Twn = Twf * Tfn;
	// get scaled object scales/propotions
	dims.push_back(max_r*2);
	dims.push_back(max_g*2);
	dims.push_back(max_b*2);
	// get final uncertainty for the centroid
	Eigen::Matrix3d R_opt = Twn.topLeftCorner(3,3).cast<double>();
	Eigen::Matrix3d sR_opt = ((double)scale) * R_opt;
	Sigma_t = avg_Sigma_bp + sR_opt*Sigma_NOCS*sR_opt.transpose();
	// std::cout << "sR_opt*Sigma_NOCS*sR_opt:" << std::endl
	// 		  << sR_opt*Sigma_NOCS*sR_opt.transpose() << std::endl;
	// 0.00760903  2.46034e-10 -4.20294e-11
	// 2.46034e-10   0.00760903 -3.51147e-10
	// -4.20294e-11 -3.51147e-10   0.00760902
	// std::cout << "Sigma_t:" << std::endl
	// 		  << Sigma_t << std::endl;

	return pt_cloud_wcs;
}

Eigen::Matrix4f RgbdFrame::EstimateSimilarityTransform(std::vector<Eigen::Vector3f> source, 
													   std::vector<Eigen::Vector3f> target,
													   float& scale)
{
	// RANSAC to get better pose estimation
	size_t max_iter = 100, num_pt = 7;
	float passThre = 0.01,
		  bestResidual = 1000000;
	Eigen::Matrix4f objPose = Eigen::Matrix4f::Identity();
	std::vector<Eigen::Vector3f> vBestSource, vBestTarget;
	int max_idx = source.size();
	if(max_idx < num_pt){
		std::cout << " ! Minimum " << num_pt << " points needed!!! Return Zero Matrix." << std::endl;
		return Eigen::Matrix4f::Zero();
	}
	srand(time(0));
	for(size_t i=0; i<max_iter; ++i)
	{
		// Randomly pick N points to estimate the pose and test on the rest
		//- 1 generate random indices
		std::set<int> sInd;
		std::vector<Eigen::Vector3f> sub_source, sub_target;
		int tmpCount = 0;
		while(sInd.size() < num_pt){
			int ranInd = rand()%max_idx;
			sInd.insert(ranInd);
			tmpCount++;
			if(tmpCount > sInd.size())
			{
				// duplicate number found
				tmpCount--;
				continue;
			}
			sub_source.push_back(source[ranInd]);
			sub_target.push_back(target[ranInd]);
		}

		//- 2 get the estimated pose
		float tmpScale;
		Eigen::Matrix4f tmpPose = SimilarityHorn(sub_source, sub_target, tmpScale);
		Eigen::Matrix3f tmpRot = tmpPose.topLeftCorner(3,3);
		Eigen::Vector3f tmpTran = tmpPose.topRightCorner(3,1);

		//- 3 evaluate the model
		float loss = 0;
		std::vector<Eigen::Vector3f> vInlierS, vInlierT;
		for(size_t j=0; j<max_idx; ++j)
		{
			Eigen::Vector3f evDiff = target[j] - tmpScale * tmpRot * source[j] - tmpTran;
			float dist = evDiff.norm();
			if(dist < passThre){
				vInlierS.push_back(source[j]);
				vInlierT.push_back(target[j]);
				// loss += passThre;
			} else {
				// loss += dist;
				loss += 1;
			}
		}
		// compare the loss
		if(loss < bestResidual){
			// store the model
			bestResidual = loss;
			vBestSource = vInlierS;
			vBestTarget = vInlierT;
		}
		
		// std::cout << vInlierT.size() << " - " << loss << std::endl;
	}
	// std::cout << "   best residual = " << bestResidual << " with "
	// 		  << vBestTarget.size() << " inliers out of "
	// 		  << target.size() << " points." << std::endl;

	//- 4 get the final pose
	if(vBestTarget.size() > target.size()/10 && vBestTarget.size() >= 7){
		return SimilarityHorn(vBestSource, vBestTarget, scale);
	} else {
		std::cout << " ! No Enough Inliers (At least 10%/7-pts needed). Return Zero Matrix." << std::endl;
		// return SimilarityHorn(source, target, scale);
		return Eigen::Matrix4f::Zero();
	}
	
}
//- One MAIN change compare to the original code:
//- the algorithm for pose calculation is changed from "Umeyama1991" to "Horn1988"
//- The scale value from Umeyama algorithm is too large
Eigen::Matrix4f RgbdFrame::SimilarityHorn(std::vector<Eigen::Vector3f> source, 
										  std::vector<Eigen::Vector3f> target,
										  float& sc)
{
	//- source is the 3d model in the NOCS (normalized object coordinate space)
	//- target is the 3d points obtained from depth map
	//- target = R * source + t
	//- R, t are the pose of the object w.r.t. the normalized model
	Eigen::Matrix4f estimate = Eigen::Matrix4f::Identity();
	Eigen::Vector3f src_mean = Eigen::Vector3f::Zero();
	Eigen::Vector3f dst_mean = Eigen::Vector3f::Zero();
	int no_inliers = 0;

	Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
	for(size_t i=0; i<source.size(); ++i)
	{
		no_inliers++;
		src_mean += source[i];
		dst_mean += target[i];
		M += source[i] * target[i].transpose();
	}

	// compute the centroid
	src_mean /= no_inliers;
	dst_mean /= no_inliers;
	M -= no_inliers * (src_mean * dst_mean.transpose());

	// compute the variance of the source
	float src_var = 0, tar_var = 0;
	for(size_t i=0; i<source.size(); ++i)
	{
		src_var += (source[i] - src_mean).squaredNorm();
		tar_var += (target[i] - dst_mean).squaredNorm();
	}
	src_var /= no_inliers;
	tar_var /= no_inliers;

	const auto svd = M.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	const auto SingVals = svd.singularValues();		// 3x1 matrix
    const auto MatU = svd.matrixU();				// 3x3 matrix
    const auto MatV = svd.matrixV();
	// const auto u1 = MatU.topLeftCorner(3, 1);
    // const auto u2 = MatU.topLeftCorner(3, 2).topRightCorner(3, 1);
    // const auto u3 = MatU.topRightCorner(3, 1);
	
	// rotation
	Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
	Eigen::Matrix3f R = MatV * S * MatU.transpose();
	if(R.determinant() < 0)
    {
        // std::cout << "Validating Rotation Matrx: " << R.determinant() << std::endl;
        S(2, 2) = -1;
        R = MatV * S * MatU.transpose();
    }

	// std::cout << R.determinant() << " - ";

	// scale
	// float sc = SingVals.sum()/src_var;
	if(src_var == 0)
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Something wrong, this shouldn't be 0" << std::endl;
	sc = std::sqrt(tar_var/src_var);
	// sc = 1;
	// translation
	Eigen::Vector3f t = dst_mean - sc * R * src_mean;

	// std::cout << R << std::endl;
	// std::cout << sc << std::endl;
	// std::cout << t << std::endl;

	//-! Apply scale on the points rather than on the rotation matrix
	// estimate.topLeftCorner(3,3) = sc * R;
	estimate.topLeftCorner(3,3) = R;
	estimate.topRightCorner(3,1) = t;

	// Eigen::Matrix3f sR = sc*R;
	// std::cout << sR.determinant() << std::endl;

	return estimate;
}

// void GeometricRefinement(float lamb, float tao, int win_size);
void RgbdFrame::FuseMasks(cv::Mat edge, int thre)
{
	std::clock_t start = std::clock();
	mEdge = edge.clone();
	// find connected components, 4-connected
	nConComps = cv::connectedComponentsWithStats(edge, mLabeled, mStats, mCentroids, 4);

	// fuse masks
	int nMasks;
	int max_area = 0;
	int max_idx;
	cv::Mat tmpLabeled, tmpStats, tmpCentroids;
	if(numDetection > 0){
		// TRANSFER TO GPU LATER ON
		cv::Mat fusedMask = cv::Mat::zeros(row_frame, col_frame, CV_8UC1);

		for(int i_comp=1; i_comp<nConComps; i_comp++)
		{
			// filter out small blobs
			int area_blob = mStats.at<int>(i_comp, cv::CC_STAT_AREA);
			if(area_blob < thre){
				continue;
			}
			if(area_blob > max_area && (mStats.at<int>(i_comp, cv::CC_STAT_LEFT) 
				+ mStats.at<int>(i_comp, cv::CC_STAT_LEFT)) > 0)
			{
				max_area = area_blob;
				max_idx = i_comp;
			}

			cv::Mat one_blob = (mLabeled==i_comp)/255; // binary matrix of 0/1
			
			for(int i_detect=0; i_detect<numDetection; i_detect++)
			{
				cv::Mat one_mask = (mask==vLabels[i_detect])/255; // binary
				int area_mask = cv::sum(one_mask)[0];

				cv::Mat overlap;
				cv::bitwise_and(one_blob, one_mask, overlap);
				int area_overlap = cv::sum(overlap)[0];
				// cv::Mat uunionn = ()
				
				// accept the overlap area as an object if iou is larger than a threshold
				// if(area_overlap >= area_mask*0.7 || area_overlap >= area_blob*0.7){
				if(area_overlap >= area_mask*0.05){
					if(area_overlap > area_blob * 0.7){
						// cv::Mat union_tmp;
						// cv::bitwise_or(overlap, one_blob, union_tmp);
						overlap = one_blob.clone();
					}
					// meaning blob is inside the mask, asign the blob to the mask
					fusedMask += overlap*vLabels[i_detect];
					// surjective mapping, one blob can have a only one mask map
					break;
				}
			}
		}

		mask = fusedMask.clone();

		// debug
		// cv::Mat bgr;
		// cv::cvtColor(image, bgr, CV_RGB2BGR);
		// cv::imwrite("/home/yohann/detected.png", bgr);
		// cv::imwrite("/home/yohann/edge.png", edge);
		// cv::imwrite("/home/yohann/connected_components.png", mLabeled);
		// cv::imwrite("/home/yohann/mask_fused.png", fusedMask*255);
		// cv::imwrite("/home/yohann/world_plane.png", world_plane*255);
		// // draw mask countors
		// std::vector<std::vector<cv::Point> > contours;
		// std::vector<cv::Vec4i> hierarchy;
		// cv::findContours(fusedMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		// cv::Scalar color(255, 0, 0);
		// cv::drawContours(image_backup, contours, -1, color, 4);
		// cv::cvtColor(image_backup, bgr, CV_RGB2BGR);
		// cv::imwrite("/home/yohann/final_detected.png", bgr);
		// // usleep(500000);
	} 

	std::cout << "#### Fuse masks takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << " seconds" << std::endl;
}
cv::Scalar RgbdFrame::CalculateColor(long int label)
{
	cv::Scalar color(
			palette[label][0], 
			palette[label][1], 
			palette[label][2]
		);
	return color;
}

cv::Mat RgbdFrame::Array2Mat(int* aMask)
{
	// cv::Mat mMask(row_frame, col_frame, CV_32SC1, aMask, sizeof(int)*col_frame);
	cv::Mat mMask(row_frame, col_frame, CV_32SC1);
	std::memcpy(mMask.data, aMask, row_frame*col_frame*sizeof(int));
	// cv::imwrite("/home/yohann/mask.jpg", mMask*255);
	// Scale will cause the value invalid for accessing, somehow
	// cv::Mat mScaleMask(row_frame, col_frame, CV_8UC1);
	// mMask.convertTo(mScaleMask, CV_8UC1);
	// // cv::imwrite("/home/yohann/mask_scale.jpg", mScaleMask*255);

	return mMask;
}
cv::Mat RgbdFrame::Array2Mat(float* aMask)
{
	// either way could get the job done blew
	// cv::Mat mMask(row_frame, col_frame, CV_32FC1, aMask, sizeof(float)*col_frame);
	cv::Mat mMask(row_frame, col_frame, CV_32FC1);
	std::memcpy(mMask.data, aMask, row_frame*col_frame*sizeof(float));
	return mMask;
}
// cv::Mat RgbdFrame::Array2Mat(float* aObj, int num)
// {
// 	// std::cout << " 1 Axis in array: " << std::endl;
//     // for(int i=0; i<12; i++){
//     //     std::cout << *(aObj+i) << ", ";
//     // }
// 	// std::cout << std::endl;
// 	cv::Mat mObj(3, num, CV_32FC1, aObj);
// 	// std::cout << "display values in opencv mat" << std::endl;
// 	// std::cout << mObj << std::endl;
	
// 	return mObj;
// }

void RgbdFrame::UpdateCentMatrix(std::vector<std::shared_ptr<Object3d>> map_obj,
                        		 std::vector<std::pair<int, int>> v_best_map_cub_labidx)
{
	// std::cout << map_obj.size() << " - " << v_best_map_cub_labidx.size() << std::endl;

	// // reset cent matrix
	// cent_matrix = cv::Mat::zeros(cv::Size(3, 30), CV_32FC1);	// 6*5 rows and 3 cols
	// // get current frame pose
	// Eigen::Matrix3d R = pose.matrix().topLeftCorner(3,3);
	// Eigen::Vector3d t = pose.matrix().topRightCorner(3,1);
	// // update the centroids in the map, row id = object id
	// for(size_t i=0; i<map_obj.size(); ++i)
	// {
	// 	for(size_t j=0; j<v_best_map_cub_labidx.size(); ++j)
	// 	{
	// 		if(map_obj[i]->label == v_best_map_cub_labidx[j].first && v_best_map_cub_labidx[j].second >= 0)
	// 		{
	// 			int row = map_obj[i]->label - 1;
	// 			// Eigen::Vector3d pos = map_obj[i]->pos;	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// 			Eigen::Vector3d pos = map_obj[i]->v_all_cuboids[v_best_map_cub_labidx[j].second]->centroid;	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// 			// from world coordinate frame to camera coordinate frame
	// 			pos = R.transpose() * (pos - t);
	// 			cent_matrix.at<float>(row, 0) = pos(0);
	// 			cent_matrix.at<float>(row, 1) = pos(1);
	// 			cent_matrix.at<float>(row, 2) = pos(2);
	// 			// std::cout << pos(0) << ", "  << pos(1) << ", "  << pos(2) << std::endl;
	// 		}
	// 	}
	// }
}

void RgbdFrame::ReprojectMapInliers(std::vector<std::shared_ptr<Object3d>> map_obj,
                        			std::vector<std::pair<int, std::pair<int, int>>> v_inlier_pairs)
                        			// std::vector<std::pair<int, int>> v_best_map_cub_labidx)
{
	// std::cout << map_obj.size() << " map objects with " << v_inlier_pairs.size()
	// 		  << "/" << v_best_map_cub_labidx.size() << " inliers\n";

	// reset cent matrix
	cent_matrix = cv::Mat::zeros(cv::Size(3, 30), CV_32FC1);	// 6*5 rows and 3 cols
	// get current frame pose
	Eigen::Matrix3d R = pose.matrix().topLeftCorner(3,3);
	Eigen::Vector3d t = pose.matrix().topRightCorner(3,1);

	// update centroids values
	for(size_t i=0; i<v_inlier_pairs.size(); ++i)
	{
		int cub_idx = v_inlier_pairs[i].second.second;
		if(cub_idx >= 0)
		{
			int map_obj_idx = v_inlier_pairs[i].second.first;
			int map_obj_label = map_obj[map_obj_idx]->label;
			int row = map_obj_label - 1;
			while(cent_matrix.at<float>(row, 0)!=0 || 
				cent_matrix.at<float>(row, 1)!=0 || 
				cent_matrix.at<float>(row, 2)!=0)
			{ row += 6; }
			if(row >= 30)
				continue;

			Eigen::Vector3d pos = map_obj[map_obj_idx]->v_all_cuboids[cub_idx]->centroid;
			pos = R.transpose() * (pos - t);
			cent_matrix.at<float>(row, 0) = pos(0);
			cent_matrix.at<float>(row, 1) = pos(1);
			cent_matrix.at<float>(row, 2) = pos(2);
		}
		else 
		{
			// std::cout << i << " is outlier. Skipped." << std::endl;
		}
		
	}
}

void RgbdFrame::UpdateFrameInliers(std::vector<std::pair<int, std::pair<int, int>>> v_inlier_pairs)
                        		//    std::vector<std::pair<int, int>> v_best_map_cub_labidx)
{
	// reset cent matrix
	cent_matrix = cv::Mat::zeros(cv::Size(3, 30), CV_32FC1);	// 6*5 rows and 3 cols
	
	// update inliers centroids
	for(size_t i=0; i<v_inlier_pairs.size(); ++i)
	{
		int cub_idx = v_inlier_pairs[i].second.second;
		if(cub_idx >= 0)
		{
			int frm_obj_idx = v_inlier_pairs[i].first;
			int frm_obj_label = vObjects[frm_obj_idx]->label;
			int row = frm_obj_label - 1;
			while(cent_matrix.at<float>(row, 0)!=0 || 
				cent_matrix.at<float>(row, 1)!=0 || 
				cent_matrix.at<float>(row, 2)!=0)
			{ row += 6; }
			if(row >= 30)
				continue;

			Eigen::Vector3d pos = vObjects[frm_obj_idx]->pos;
			cent_matrix.at<float>(row, 0) = pos(0);
			cent_matrix.at<float>(row, 1) = pos(1);
			cent_matrix.at<float>(row, 2) = pos(2);
		}
		else
		{
			// std::cout << i << " is outlier, skipped." << std::endl;
		}
		
	}
}


} // namespace fusion