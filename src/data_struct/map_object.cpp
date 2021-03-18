#include "data_struct/map_object.h"

namespace fusion
{

Object3d::Object3d(std::shared_ptr<Cuboid3d> cub)
{
    label = cub->label;
    observation_count = cub->observation;
    
    pos = cub->centroid;
    cov = cub->cov;
    primary_cuboid_idx = 0;
    
    v_all_cuboids.push_back(std::move(cub));

    cuboid_idx = -1;
}

Object3d::Object3d(std::shared_ptr<Object3d> ref_obj)
{
    label = ref_obj->label;
    observation_count = ref_obj->observation_count;
    
    pos = ref_obj->pos;
    cov = ref_obj->cov;
    primary_cuboid_idx = ref_obj->primary_cuboid_idx;

    v_all_cuboids = ref_obj->v_all_cuboids;
    
    cuboid_idx = ref_obj->cuboid_idx;
}

void Object3d::copyFrom(std::shared_ptr<Object3d> ref_obj)
{
    label = ref_obj->label;
    observation_count = ref_obj->observation_count;
    
    pos = ref_obj->pos;
    cov = ref_obj->cov;
    primary_cuboid_idx = ref_obj->primary_cuboid_idx;

    v_all_cuboids = ref_obj->v_all_cuboids;
    
    cuboid_idx = ref_obj->cuboid_idx;
}

// bool Object3d::bbox3d_overlap(Eigen::Vector3d &cent, std::vector<float> &dim, float s)
// {
//     // loop through all cuboids belong to this object
//     bool bOverlapWithObject = false;
//     float ratio_threshold = 0.5,
//           overlap_map_ratio_max = 0,
//           overlap_frame_ratio_max = 0;
//     for(size_t idx=0; idx<v_all_cuboids.size(); ++idx)
//     {
//         std::vector<float> sDim1, sDim2;
//         float rMap, rFrame;
//         for(size_t i=0; i<3; ++i)
//         {
//             sDim1.push_back(dim[i]*s);
//             sDim2.push_back(v_all_cuboids[idx]->dims[i] * v_all_cuboids[idx]->scale);
//         }
//         std::vector<float> overlap_min, overlap_max;
//         bool bOverlapWithCuboid = true;
//         for(size_t i=0; i<3; ++i)
//         {
//             overlap_min.push_back( std::max(cent(i)-sDim1[i]/2, v_all_cuboids[idx]->centroid(i)-sDim2[i]/2) );
//             overlap_max.push_back( std::min(cent(i)+sDim1[i]/2, v_all_cuboids[idx]->centroid(i)+sDim2[i]/2) );
//             if(overlap_min[i] >= overlap_max[i]){
//                 // no overlap
//                 bOverlapWithCuboid = false;
//                 break;
//             }
//         }
//         if(bOverlapWithCuboid){
//             float overlap_volume = (overlap_max[0]-overlap_min[0])*(overlap_max[1]-overlap_min[1])*(overlap_max[2]-overlap_min[2]);
//             float map_volume = sDim1[0] * sDim1[1] * sDim1[2];
//             rMap = overlap_volume/map_volume;
//             float frame_volume = sDim2[0] * sDim2[1] * sDim2[2];
//             rFrame = overlap_volume/frame_volume;
//             bOverlapWithObject = true;
            
//             // find the cuboid with max overlap
//             if(rMap > overlap_map_ratio_max){
//                 overlap_map_ratio_max = rMap;
//                 overlap_frame_ratio_max = rFrame;
//                 cuboid_idx = idx;
//             }
//         }
//     }//-it_cub

//     // if criteria met, update that cuboid, else store as new cuboid
//     if(overlap_map_ratio_max < ratio_threshold && overlap_frame_ratio_max < ratio_threshold){
//         // std::cout << overlap_map_ratio_max << " - " << overlap_frame_ratio_max << std::endl;
//         // std::cout << "!!!!!!!!!! Overlap volume too small, add new cuboid." << std::endl;
//         cuboid_idx = -1;
//     }

//     return bOverlapWithObject;
// }

float Object3d::bbox3d_overlap(Eigen::Vector3d &cent, std::vector<float> &dim, float s)
{
    // loop through all cuboids belong to this object
    float max_IoU = 0.;
    for(size_t idx=0; idx<v_all_cuboids.size(); ++idx)
    {
        std::vector<float> sDim1, sDim2;
        float IoU;
        for(size_t i=0; i<3; ++i)
        {
            sDim1.push_back(dim[i]*s);
            sDim2.push_back(v_all_cuboids[idx]->dims[i] * v_all_cuboids[idx]->scale);
        }
        std::vector<float> overlap_min, overlap_max;
        bool bOverlapWithCuboid = true;
        for(size_t i=0; i<3; ++i)
        {
            overlap_min.push_back( std::max(cent(i)-sDim1[i]/2, v_all_cuboids[idx]->centroid(i)-sDim2[i]/2) );
            overlap_max.push_back( std::min(cent(i)+sDim1[i]/2, v_all_cuboids[idx]->centroid(i)+sDim2[i]/2) );
            if(overlap_min[i] >= overlap_max[i]){
                // no overlap
                bOverlapWithCuboid = false;
                break;
            }
        }
        if(bOverlapWithCuboid){
            float overlap_volume = (overlap_max[0]-overlap_min[0])*(overlap_max[1]-overlap_min[1])*(overlap_max[2]-overlap_min[2]);
            float map_volume = sDim1[0] * sDim1[1] * sDim1[2];
            float frame_volume = sDim2[0] * sDim2[1] * sDim2[2];
            IoU = overlap_volume / (map_volume+frame_volume-overlap_volume);
            
            // bOverlapWithObject = true;
            
            // find the cuboid with max overlap
            if(IoU > max_IoU){
                max_IoU = IoU;
                cuboid_idx = idx;
            }
        }
    }//-it_cub

    // if criteria met, update that cuboid, else store as new cuboid
    if(max_IoU < 0.3){
        std::cout << "\nMax IoU = " << max_IoU << "!!!!!!!!!! Overlap volume too small, add new cuboid." << std::endl;
        cuboid_idx = -1;
    }

    return max_IoU;

}

void Object3d::update_object(std::shared_ptr<Cuboid3d> obs)
{
    // changing the criterion here from IoU to Mahalanobis distance (MD)
    // loop through all the centroids of this object and compute the corresponding MD
    float chi2 = 16.2662;
    float min_MD2 = 16.2662;        // 12.8382 16.2662
    std::vector<int> vIdx;
    cuboid_idx = -1;
    // std::cout << "\nFrame Object ID " << obs->label << std::endl;
    for(size_t idx=0; idx<v_all_cuboids.size(); ++idx)
    {
        Eigen::Vector3d obs_cent = obs->centroid;
        Eigen::Vector3d exp_cent = v_all_cuboids[idx]->mean;
        Eigen::Matrix3d covariance = v_all_cuboids[idx]->cov;
        Eigen::Vector3d diff_cent = obs_cent - exp_cent;

        double MD2 = diff_cent.transpose() * covariance.inverse() * diff_cent;
        
        // if(obs->label == 5){
            // std::cout << "Map Cuboid" << idx << std::endl;
            // std::cout << obs->centroid << "\n"
            //         << v_all_cuboids[idx]->mean << "\n" 
            //         << v_all_cuboids[idx]->cov << "\n"; 
                    // << v_all_cuboids[idx]->cov.inverse() << "\n"
                    // << v_all_cuboids[idx]->cov*v_all_cuboids[idx]->cov.inverse() << std::endl;
            // std::cout << "Number of cent stored: " << v_all_cuboids[idx]->vCentroids.size() << std::endl;
            // std::cout << "The squared mahalanobis distance is " << MD2 << std::endl;
        // }

        if (MD2 < chi2)
        {
            // std::cout << "MATCHED!!" << std::endl;
            vIdx.push_back(idx);
            if(MD2 < min_MD2){
                cuboid_idx = idx;
                min_MD2 = MD2;
            }
        }
    }
    // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << vIdx.size() << std::endl;


    // merge 2 map cuboid if new centroid fall into the confident zone of both
    int matched_cub = vIdx.size();
    if(matched_cub == 0)
    {
        // store the new cuboid to the list
        v_all_cuboids.push_back(std::move(obs));
    }
    else
    {
        // First update the first map cub with the observation
        update_cuboid(v_all_cuboids[vIdx[0]], obs);

        // Second, merge the rest into the fisrt cuboid, in reversed order
        for(size_t i=matched_cub-1; i>0; --i)
        {
            update_cuboid(v_all_cuboids[vIdx[0]], v_all_cuboids[vIdx[i]]);
            v_all_cuboids.erase(v_all_cuboids.begin()+vIdx[i]);
        }
    }
    
    // // update the object with that observed cuboid
    // if(cuboid_idx >= 0)
    // {
    //     //- Update the gaussian probability model
    //     // update mean CONSIDER MERGE MEAN UPDATE WITH CENTROID UPDATE
    //     v_all_cuboids[cuboid_idx]->vCentroids.push_back(obs->centroid);
    //     int count = v_all_cuboids[cuboid_idx]->vCentroids.size();
    //     v_all_cuboids[cuboid_idx]->mean = Eigen::Vector3d::Zero();
    //     for(size_t i=0; i<count; ++i){
    //         v_all_cuboids[cuboid_idx]->mean += v_all_cuboids[cuboid_idx]->vCentroids[i];
    //     }
    //     v_all_cuboids[cuboid_idx]->mean /= count;
    //     // update covariance
    //     if(count > 9)
    //     {
    //         Eigen::Matrix3d tmp_cov = Eigen::Matrix3d::Zero();
    //         // v_all_cuboids[cuboid_idx]->cov = Eigen::Matrix3d::Zero();
    //         for(size_t i=0; i<count; ++i)
    //         {
    //             Eigen::Vector3d diff = v_all_cuboids[cuboid_idx]->vCentroids[i] - 
    //                             v_all_cuboids[cuboid_idx]->mean;
    //             tmp_cov += diff * diff.transpose();
    //             // if(obs->label == 5)
    //             // {
    //             //     std::cout << v_all_cuboids[cuboid_idx]->vCentroids[i](0) << ", "
    //             //               << v_all_cuboids[cuboid_idx]->vCentroids[i](1) << ", "
    //             //               << v_all_cuboids[cuboid_idx]->vCentroids[i](2) << " - "
    //             //               << diff(0) << ", " << diff(1) << ", " << diff(2) << std::endl;
    //             // }
    //         }
    //         tmp_cov /= (count-1);
    //         Eigen::FullPivLU<Eigen::Matrix3d> lu(tmp_cov);
    //         if(lu.rank() == 3)
    //             v_all_cuboids[cuboid_idx]->cov = tmp_cov;
    //         // if(obs->label == 5)
    //         // {
    //         //     std::cout << "Updated mean and cov are: \n"
    //         //               << v_all_cuboids[cuboid_idx]->mean(0) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->mean(1) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->mean(2) << std::endl
    //         //               << v_all_cuboids[cuboid_idx]->cov(0, 0) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->cov(0, 1) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->cov(0, 2) << "\n"
    //         //               << v_all_cuboids[cuboid_idx]->cov(1, 0) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->cov(1, 1) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->cov(1, 2) << "\n"
    //         //               << v_all_cuboids[cuboid_idx]->cov(2, 0) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->cov(2, 1) << ", "
    //         //               << v_all_cuboids[cuboid_idx]->cov(2, 2) << "\n";
    //         // }
    //         std::cout << "Number of cent stored: " << count << std::endl;
    //         // std::cout << "Rank of the covariance is " << lu.rank() << std::endl;
    //     }
    //     // double sigX=0, 
    //     //        sigY=0, 
    //     //        sigZ=0;
    //     // v_all_cuboids[cuboid_idx]->cov = Eigen::Matrix3d::Identity();
    //     // for(size_t i=0; i<count; ++i){
    //     //     Eigen::Vector3d diff = v_all_cuboids[cuboid_idx]->vCentroids[i] - 
    //     //                         v_all_cuboids[cuboid_idx]->mean;
    //     //     sigX += diff(0) * diff(0);
    //     //     sigY += diff(1) * diff(1);
    //     //     sigZ += diff(2) * diff(2);
    //     // }
    //     // v_all_cuboids[cuboid_idx]->cov(0,0) = sigX/count;
    //     // v_all_cuboids[cuboid_idx]->cov(1,1) = sigY/count;
    //     // v_all_cuboids[cuboid_idx]->cov(2,2) = sigZ/count;
    //     //- Using simple average over all observations to update all the parameters 
    //     double pre_weight = double(v_all_cuboids[cuboid_idx]->observation),
    //            cur_weight = double(obs->observation),
    //            sum_weight = pre_weight + cur_weight;
    //     // tranlsation
    //     Eigen::Matrix4d pre_pose = v_all_cuboids[cuboid_idx]->pose,
    //                     cur_pose = obs->pose;
    //     Eigen::Matrix3d pre_rot = pre_pose.topLeftCorner(3, 3),
    //                     cur_rot = cur_pose.topLeftCorner(3, 3);
    //     Eigen::Vector3d new_trans = (pre_weight * pre_pose.topRightCorner(3,1) + 
    //                                 cur_weight * cur_pose.topRightCorner(3, 1)) / sum_weight;
    //     v_all_cuboids[cuboid_idx]->centroid = new_trans;
    //     // rotation
    //     Eigen::Quaterniond pre_quat(pre_rot),
    //                         cur_quat(cur_rot);
    //     Eigen::Vector4d pre_vQuat(pre_quat.w(), pre_quat.x(), pre_quat.y(), pre_quat.z()),
    //                     cur_vQuat(cur_quat.w(), cur_quat.x(), cur_quat.y(), cur_quat.z());
    //     Eigen::Matrix4d A = (pre_weight*pre_vQuat*pre_vQuat.transpose() + 
    //                         cur_weight*cur_vQuat*cur_vQuat.transpose()) / sum_weight;
    //     Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(A);
    //     Eigen::Vector4d new_vQuat = eigensolver.eigenvectors().col(3);
    //     Eigen::Quaterniond new_quat(new_vQuat(0), new_vQuat(1), new_vQuat(2), new_vQuat(3));
    //     // pose
    //     Eigen::Matrix4d new_pose = Eigen::Matrix4d::Identity();
    //     new_pose.topLeftCorner(3, 3) = new_quat.toRotationMatrix();
    //     new_pose.topRightCorner(3, 1) = new_trans;
    //     v_all_cuboids[cuboid_idx]->pose = new_pose;
    //     // scale
    //     float new_scale = (pre_weight*v_all_cuboids[cuboid_idx]->scale + cur_weight*obs->scale) / sum_weight;
    //     v_all_cuboids[cuboid_idx]->scale = new_scale;
    //     // dims --- Potential Problem: principal axis changes when view-angles changes
    //     float max_r = (pre_weight * v_all_cuboids[cuboid_idx]->dims[0] + cur_weight * obs->dims[0]) / sum_weight;
    //     float max_g = (pre_weight * v_all_cuboids[cuboid_idx]->dims[1] + cur_weight * obs->dims[1]) / sum_weight;
    //     float max_b = (pre_weight * v_all_cuboids[cuboid_idx]->dims[2] + cur_weight * obs->dims[2]) / sum_weight;
    //     v_all_cuboids[cuboid_idx]->dims[0] = max_r;
    //     v_all_cuboids[cuboid_idx]->dims[1] = max_g;
    //     v_all_cuboids[cuboid_idx]->dims[2] = max_b;
    //     //- Compute new corners and axes for display 
    //     max_r *= new_scale;
    //     max_g *= new_scale;
    //     max_b *= new_scale;
    //     Eigen::Matrix<float, 3, 4> xyz_axis;	// in the order of origin, z, y, x
    //     xyz_axis << 0, 0, 0, max_r/2,
    //                 0, 0, max_g/2, 0,
    //                 0, max_b/2, 0, 0;
    //     Eigen::Matrix<float, 3, 8> bbox_3d;
    //     bbox_3d << max_r/2, max_r/2, -max_r/2, -max_r/2, max_r/2, max_r/2, -max_r/2, -max_r/2,
    //                 max_g/2, max_g/2, max_g/2, max_g/2, -max_g/2, -max_g/2, -max_g/2, -max_g/2,
    //                 max_b/2, -max_b/2, max_b/2, -max_b/2, max_b/2, -max_b/2, max_b/2, -max_b/2;
    //     // find cuboid corners and axes in world coordinate system
    //     cv::Mat corners(3, 8, CV_32FC1), 
    //             axis(3, 4, CV_32FC1);
    //     Eigen::Matrix3f Rwn = new_pose.cast<float>().topLeftCorner(3,3);
    //     Eigen::Vector3f twn = new_pose.cast<float>().topRightCorner(3,1);
    //     for(size_t c=0; c<8; ++c)
    //     {
    //         if(c<4){
    //             // Normalized Coordinate -> Camera Coordinate -> World Coordinate
    //             Eigen::Vector3f ax = Rwn * xyz_axis.col(c) + twn;
    //             for (size_t r = 0; r < 3; r++)
    //             {
    //             axis.at<float>(r,c) = ax(r);
    //             }
    //         }
    //         Eigen::Vector3f cor = Rwn * bbox_3d.col(c) + twn;
    //         for (size_t r = 0; r < 3; r++)
    //         {
    //             corners.at<float>(r,c) = cor(r);
    //         }
    //     }
    //     v_all_cuboids[cuboid_idx]->cuboid_corner_pts = corners.clone();
    //     v_all_cuboids[cuboid_idx]->axes = axis.clone();  
    //     //- Update confidence & obervations
    //     // P(X|x1,...,xn) = P(x1|X)...P(x2|X)/normalize_term
    //     v_all_cuboids[cuboid_idx]->update_confidence(obs->all_class_confidence);
    //     v_all_cuboids[cuboid_idx]->observation += obs->observation;
    //     // std::cout << v_all_cuboids[cuboid_idx]->label << " - " 
    //     //           << v_all_cuboids[cuboid_idx]->confidence << " - "
    //     //           << v_all_cuboids[cuboid_idx]->observation << std::endl;
    // }
    // else
    // {
    //     // std::cout << "!!!!!!!!!! Add new cuboid based on overlap ratio to the list." << std::endl;
    //     // std::cout << "Object " << this->label << "-" 
    //     //           << this->observation_count << "/" << this->v_all_cuboids.size() 
    //     //           << "; label-" << obs->label << std::endl;
    //     // store the new cuboid to the list
    //     // std::shared_ptr<Cuboid3d> cub(new Cuboid3d(obs)); // remove after remove vCuboids list
    //     v_all_cuboids.push_back(std::move(obs));
    // }

    // after updating, choose the most confident one to display
    float max_confidence = 0.; // computed as confidence*observation
    int sum_observation = 0;
    for(size_t idx=0; idx < v_all_cuboids.size(); ++idx)
    {
        sum_observation += v_all_cuboids[idx]->observation;
        float tmp_confidence = v_all_cuboids[idx]->confidence * v_all_cuboids[idx]->observation;
        if(tmp_confidence > max_confidence)
        {
            max_confidence = tmp_confidence;
            primary_cuboid_idx = idx;
        }
    }

    // update total count and new pos
    pos = v_all_cuboids[primary_cuboid_idx]->centroid;
    cov = v_all_cuboids[primary_cuboid_idx]->cov;
    observation_count = sum_observation;
}

void Object3d::update_cuboid(std::shared_ptr<Cuboid3d> cub1, std::shared_ptr<Cuboid3d> cub2)
{
    // update the matched cub with new observation
    //- Update the gaussian probability model
    // update mean CONSIDER MERGE MEAN UPDATE WITH CENTROID UPDATE
    cub1->vCentroids.insert(cub1->vCentroids.end(), cub2->vCentroids.begin(), cub2->vCentroids.end());
    int count = cub1->vCentroids.size();
    cub1->mean = Eigen::Vector3d::Zero();
    cub1->vScales.insert(cub1->vScales.end(), cub2->vScales.begin(), cub2->vScales.end());
    double scale_avg = 0;
    for(size_t i=0; i<count; ++i){
        cub1->mean += cub1->vCentroids[i];
        scale_avg += cub1->vScales[i];
    }
    cub1->mean /= count;
    scale_avg /= count;
    
    // update covariance
    if(count > 9)
    {
        Eigen::Matrix3d tmp_cov = Eigen::Matrix3d::Zero();
        // cub1->cov = Eigen::Matrix3d::Zero();
        double tmp_sigma = 0;
        for(size_t i=0; i<count; ++i)
        {
            Eigen::Vector3d diff = cub1->vCentroids[i] - cub1->mean;
            tmp_cov += diff * diff.transpose();
            
            tmp_sigma += (cub1->vScales[i]-scale_avg)*(cub1->vScales[i]-scale_avg);
        }
        tmp_cov /= (count-1);
        Eigen::FullPivLU<Eigen::Matrix3d> lu(tmp_cov);
        if(lu.rank() == 3)
            cub1->cov = tmp_cov;
        // std::cout << "Rank of the covariance is " << lu.rank() << std::endl;
        
        cub1->sigma_scale = tmp_sigma/(count-1);
    }

    //- Using simple average over all observations to update all the parameters 
    double pre_weight = double(cub1->observation),
           cur_weight = double(cub2->observation),
           sum_weight = pre_weight + cur_weight;
    // tranlsation
    cub1->centroid = cub1->mean;
    // rotation
    Eigen::Matrix3d pre_rot = cub1->pose.topLeftCorner(3, 3),
                    cur_rot = cub2->pose.topLeftCorner(3, 3);
    Eigen::Quaterniond pre_quat(pre_rot),
                       cur_quat(cur_rot);
    Eigen::Vector4d pre_vQuat(pre_quat.w(), pre_quat.x(), pre_quat.y(), pre_quat.z()),
                    cur_vQuat(cur_quat.w(), cur_quat.x(), cur_quat.y(), cur_quat.z());
    Eigen::Matrix4d A = (pre_weight*pre_vQuat*pre_vQuat.transpose() + 
                        cur_weight*cur_vQuat*cur_vQuat.transpose()) / sum_weight;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(A);
    Eigen::Vector4d new_vQuat = eigensolver.eigenvectors().col(3);
    Eigen::Quaterniond new_quat(new_vQuat(0), new_vQuat(1), new_vQuat(2), new_vQuat(3));
    // pose
    Eigen::Matrix4d new_pose = Eigen::Matrix4d::Identity();
    new_pose.topLeftCorner(3, 3) = new_quat.toRotationMatrix();
    new_pose.topRightCorner(3, 1) = cub1->mean;
    cub1->pose = new_pose;
    // scale
    float new_scale = (pre_weight*cub1->scale + cur_weight*cub2->scale) / sum_weight;
    cub1->scale = new_scale;
    // dims --- Potential Problem: principal axis changes when view-angles changes
    float max_r = (pre_weight*cub1->dims[0] + cur_weight*cub2->dims[0]) / sum_weight;
    float max_g = (pre_weight*cub1->dims[1] + cur_weight*cub2->dims[1]) / sum_weight;
    float max_b = (pre_weight*cub1->dims[2] + cur_weight*cub2->dims[2]) / sum_weight;
    cub1->dims[0] = max_r;
    cub1->dims[1] = max_g;
    cub1->dims[2] = max_b;
    
    //- Compute new corners and axes for display 
    max_r *= new_scale;
    max_g *= new_scale;
    max_b *= new_scale;
    Eigen::Matrix<float, 3, 4> xyz_axis;	// in the order of origin, z, y, x
    xyz_axis << 0, 0, 0, max_r/2,
                0, 0, max_g/2, 0,
                0, max_b/2, 0, 0;
    Eigen::Matrix<float, 3, 8> bbox_3d;
    bbox_3d << max_r/2, max_r/2, -max_r/2, -max_r/2, max_r/2, max_r/2, -max_r/2, -max_r/2,
                max_g/2, max_g/2, max_g/2, max_g/2, -max_g/2, -max_g/2, -max_g/2, -max_g/2,
                max_b/2, -max_b/2, max_b/2, -max_b/2, max_b/2, -max_b/2, max_b/2, -max_b/2;
    // find cuboid corners and axes in world coordinate system
    cv::Mat corners(3, 8, CV_32FC1), 
            axis(3, 4, CV_32FC1);
    Eigen::Matrix3f Rwn = new_pose.cast<float>().topLeftCorner(3,3);
    Eigen::Vector3f twn = new_pose.cast<float>().topRightCorner(3,1);
    for(size_t c=0; c<8; ++c)
    {
        if(c<4){
            // Normalized Coordinate -> Camera Coordinate -> World Coordinate
            Eigen::Vector3f ax = Rwn * xyz_axis.col(c) + twn;
            for (size_t r = 0; r < 3; r++)
            {
            axis.at<float>(r,c) = ax(r);
            }
        }
        Eigen::Vector3f cor = Rwn * bbox_3d.col(c) + twn;
        for (size_t r = 0; r < 3; r++)
        {
            corners.at<float>(r,c) = cor(r);
        }
    }
    cub1->cuboid_corner_pts = corners.clone();
    cub1->axes = axis.clone();
    
    //- Update confidence & obervations
    // P(X|x1,...,xn) = P(x1|X)...P(x2|X)/normalize_term
    cub1->update_confidence(cub2->all_class_confidence);
    cub1->observation += cub2->observation;
}

Object3d::Object3d(std::string file_name, int& start_line)
{
    readFromFile(file_name, start_line);
}

void Object3d::writeToFile(std::string file_name)
{
    std::ofstream semanticfile;
    semanticfile.open(file_name+"-semantic.txt", std::ios::app);
    if (semanticfile.is_open())
    {
        semanticfile << label << "," << observation_count << "\n";
        semanticfile << primary_cuboid_idx <<  "," << v_all_cuboids.size() << "\n"
                     << pos(0) << "," << pos(1) << "," << pos(2) << "\n"
                     << cov(0,0) << "," << cov(0,1) << "," << cov(0,2) << ","
                     << cov(1,0) << "," << cov(1,1) << "," << cov(1,2) << ","
                     << cov(2,0) << "," << cov(2,1) << "," << cov(2,2) << "\n";
    }
    semanticfile.close();
    for(size_t i=0; i<v_all_cuboids.size(); ++i)
    {
        v_all_cuboids[i]->writeToFile(file_name);
    }
}

void Object3d::readFromFile(std::string file_name, int& start_line)
{

    int num_of_cuboids = 0;
    std::ifstream semanticfile(file_name+"-semantic.txt", std::ios::in);
    if (semanticfile.is_open())
    {
        std::string line;
        size_t line_idx = 0;
        int cuboid_start_line = start_line; 
        while(std::getline(semanticfile, line))
        {
            if(line_idx == start_line)
            {
                // line 1, lable, obs_count
                size_t p=line.find(",");
                label = std::stoi(line.substr(0, p));
                line.erase(0, p+1);
                p=line.find(",");
                observation_count = std::stof(line.substr(0, p));
                cuboid_start_line += 1;
                // std::cout << "Label-count: " << label << ", " << observation_count << std::endl;
                // line 2, primary_idx, cuboid size
                std::getline(semanticfile, line);
                p=line.find(",");
                primary_cuboid_idx = std::stoi(line.substr(0, p));
                line.erase(0, p+1);
                p=line.find(",");
                num_of_cuboids = std::stof(line.substr(0, p));
                cuboid_start_line += 1;
                // std::cout<< "cuboid: " << primary_cuboid_idx << ", " << num_of_cuboids << std::endl;
                // line 3 position
                std::getline(semanticfile, line);
                for(size_t idx=0; idx<3; ++idx){
                    p=line.find(",");
                    pos(idx) = std::stof(line.substr(0, p));
                    line.erase(0, p+1);
                }
                cuboid_start_line += 1;
                // std::cout << "position: " << pos(0) << ", " << pos(1) << ", " << pos(2) << std::endl;
                // line 4, covariance
                std::getline(semanticfile, line);
                for(size_t r=0; r<3; ++r){
                    for(size_t c=0; c<3; ++c){
                        p=line.find(",");
                        cov(r,c)=std::stof(line.substr(0, p));
                        line.erase(0, p+1);
                    }
                }
                cuboid_start_line += 1;
                // std::cout << cov << std::endl;
                // line 5 and on: cuboid information
                for(size_t idx=0; idx<num_of_cuboids; ++idx)
                {
                    std::shared_ptr<Cuboid3d> new_cuboid(new Cuboid3d(file_name, cuboid_start_line));
                    v_all_cuboids.push_back(std::move(new_cuboid));
                    cuboid_start_line += 9;
                }

                break;
            }
            line_idx++;
        }
    }
    semanticfile.close();
    if(num_of_cuboids > 0){
        start_line += (4 + 9*num_of_cuboids);
    } else {
        std::cout << "!!!! Something went wrong, there should be some cuboids here." << std::endl;
    }
}

}