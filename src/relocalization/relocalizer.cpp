#include "relocalization/relocalizer.h"
#include "relocalization/ransac_ao.h"
#include "tracking/cuda_imgproc.h"
#include "optimizer/graph_optimizer.h"
#include <cmath>
#include <random>

namespace fusion
{

class Optimizer;

Relocalizer::Relocalizer(const Eigen::Matrix3f intrinsic_inv) : KInv(intrinsic_inv)
{
    label_pose.push_back(3);
    label_pose.push_back(5);
    label_vec.push_back(1);
    label_vec.push_back(2);
    label_vec.push_back(4);
    label_vec.push_back(6);
}

void Relocalizer::set_target_frame(std::shared_ptr<RgbdFrame> frame)
{
    target_frame = frame;
}

void Relocalizer::compute_pose_candidates(std::vector<Sophus::SE3d> &candidates)
{
    // target_frame->pose = Sophus::SE3d();
    // std::vector<cv::KeyPoint> raw_keypoints;
    // cv::Mat raw_descriptors;

    // cv::cuda::GpuMat depth(target_frame->depth);
    // cv::cuda::GpuMat vmap_gpu, nmap_gpu;
    // backProjectDepth(depth, vmap_gpu, KInv);
    // computeNMap(vmap_gpu, nmap_gpu);

    // extractor->extract_features_surf(
    //     target_frame->image,
    //     raw_keypoints,
    //     raw_descriptors);

    // extractor->compute_3d_points(
    //     cv::Mat(vmap_gpu),
    //     cv::Mat(nmap_gpu),
    //     raw_keypoints,
    //     raw_descriptors,
    //     target_frame->cv_key_points,
    //     target_frame->descriptors,
    //     target_frame->key_points,
    //     target_frame->pose.cast<float>());

    // std::vector<std::vector<cv::DMatch>> matches;
    // matcher->match_hamming_knn(
    //     map_descriptors,
    //     target_frame->descriptors,
    //     matches, 2);

    // std::vector<cv::DMatch> list;
    // std::vector<std::vector<cv::DMatch>> candidate_matches;
    // matcher->filter_matches_ratio_test(matches, list);
    // candidate_matches.push_back(list);
    // // matcher->filter_matches_pair_constraint(target_frame->key_points, map_points, matches, candidate_matches);

    // for (const auto &match_list : candidate_matches)
    // {
    //     std::vector<Eigen::Vector3f> src_pts, dst_pts;
    //     for (const auto &match : match_list)
    //     {
    //         src_pts.push_back(map_points[match.trainIdx]->pos);
    //         dst_pts.push_back(target_frame->key_points[match.queryIdx]->pos);
    //     }

    //     std::vector<bool> outliers;
    //     Eigen::Matrix4f estimate;
    //     float inlier_ratio, confidence;
    //     PoseEstimator::RANSAC(src_pts, dst_pts, outliers, estimate, inlier_ratio, confidence);

    //     const int no_inliers = std::count(outliers.begin(), outliers.end(), false);
    //     std::cout << estimate << std::endl
    //               << no_inliers << std::endl;

    //     candidates.emplace_back(Sophus::SE3f(estimate).cast<double>());
    // }
}

// Multiple instances per object class exist.
std::vector<Eigen::Matrix4d> Relocalizer::object_data_association(std::vector<std::shared_ptr<Object3d>> frame_obj,
                                                          std::vector<std::shared_ptr<Object3d>> map_obj,
                                                          std::vector<std::vector< std::pair<int, std::pair<int, int>> >>& vv_inlier_pairs,
                                                          bool & b_enough_corresp, bool & b_recovered)
{
    // Use all configurations instead of the primary one here
    std::vector<Eigen::Matrix4d> pose_candidates;
    if(frame_obj.size() < 3){
        return pose_candidates;
    }

    //- Use the label consistency to get potential map-frame pair
    std::vector<std::pair<int, std::pair<int, int>>> v_pair_all;
    for(size_t i=0; i< frame_obj.size(); ++i)
    {
        for(size_t j=0; j<map_obj.size(); ++j)
        {
            if(frame_obj[i]->label == map_obj[j]->label)
            {
                for(size_t k=0; k<map_obj[j]->v_all_cuboids.size(); ++k)
                {
                    v_pair_all.push_back(std::make_pair(i, std::make_pair(j, k))); // first-Frm, second.first-Map, second.second-cuboid   
                    // std::cout << "Label-" << map_obj[j]->label << "(" << map_obj[j]->observation_count 
                    //       <<  "): Frame-" << i << ", Map-" << j << "-" << k << std::endl;    
                }
            }
        }
    }
    //- Construct the Adjacency matrix
    float thre = 0.9;
    const int NUM_PAIR_CANDIDATE = v_pair_all.size();
    // std::cout << " IN TOTAL " << NUM_PAIR_CANDIDATE << " correspondences found." << std::endl;

    // Use Eigenvector to solve the AM
    Eigen::MatrixXd adjMat = Eigen::MatrixXd::Zero(NUM_PAIR_CANDIDATE, NUM_PAIR_CANDIDATE);
    for (int y = 0; y < NUM_PAIR_CANDIDATE; ++y)
    {
        auto pair_i = v_pair_all[y];
        Eigen::Vector3d frm_i = frame_obj[pair_i.first]->pos;
        Eigen::Vector3d map_i = map_obj[pair_i.second.first]->v_all_cuboids[pair_i.second.second]->centroid;
        double frm_s = double(frame_obj[pair_i.first]->v_all_cuboids[0]->scale);
        double map_s = double(map_obj[pair_i.second.first]->v_all_cuboids[pair_i.second.second]->scale);
        double sigma_s = map_obj[pair_i.second.first]->v_all_cuboids[pair_i.second.second]->sigma_scale;
        double scale_dist = std::exp(-1 * (frm_s-map_s) * (frm_s-map_s) / (2 * sigma_s) );
        double scale_ratio = (frm_s > map_s) ? map_s/frm_s : frm_s/map_s;

        for (int x = 0; x < NUM_PAIR_CANDIDATE; ++x)
        {
            auto pair_j = v_pair_all[x];
            Eigen::Vector3d frm_j = frame_obj[pair_j.first]->pos;
            Eigen::Vector3d map_j = map_obj[pair_j.second.first]->v_all_cuboids[pair_j.second.second]->centroid;
            if(x==y)
            {
                // adjMat(y, x) = scale_dist;
                adjMat(y, x) = scale_ratio;
            }
            else if (y < x)
            {
                // compute the distance consistency score
                const float map_dist = float( (map_i - map_j).norm() );
                const float frm_dist = float( (frm_i - frm_j).norm() );
                // const float min_xz = std::min();
                float score = std::exp(-1 * std::fabs(map_dist - frm_dist));         
                if (std::isnan(score))
                    score = 0;
                adjMat(y, x) = score;     
            }
            else
            {
                // symmetric matrix, set directly
                adjMat(y, x) = adjMat(x, y);
            }
        } // end cols
    } // end rows
    // std::cout << "EIGEN::MAT: \n";
    // std::cout << adjMat << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(adjMat);
    // std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
    // std::cout << "Here's a matrix whose columns are eigenvectors of adjMat \n"
    //           << "corresponding to these eigenvalues:\n"
    //           << eigensolver.eigenvectors() << std::endl;
    Eigen::VectorXd xStar = eigensolver.eigenvectors().col(NUM_PAIR_CANDIDATE-1);
    // std::cout << "x*:\n" << xStar << std::endl;

    // loop through to find the optimal assignment/correspondences
    // std::vector<int> inlier_index;
    std::vector<std::pair<int, std::pair<int, int>>> v_inlier_candidates;
    std::vector<std::shared_ptr<Object3d>> vMapObj;
    std::vector<std::shared_ptr<Object3d>> vFrmObj;
    std::vector<Eigen::Vector3d> v_frame_centroids, v_map_centroids;
    std::vector<Eigen::Matrix3d> v_frame_covs, v_map_covs;
    std::vector<double> v_weight;
    // std::vector<std::pair<int, int>> v_map_cub_labidx;
    for(size_t i=0; i<NUM_PAIR_CANDIDATE; ++i)
    {
        int max_idx = -1;
        double max_val = xStar.maxCoeff(&max_idx);
        // std::cout << "Max val = " << max_val << " at " << max_idx << std::endl;
        if(max_val > 0)
        {
            // get val and check for conflicts
            // check for conflict
            int frm_idx = v_pair_all[max_idx].first;
            int map_idx = v_pair_all[max_idx].second.first;
            int cub_idx = v_pair_all[max_idx].second.second;
            int frm_label = frame_obj[frm_idx]->label;
            // v_inlier_candidates.push_back(std::make_pair(map_idx, frm_idx));
            v_inlier_candidates.push_back(v_pair_all[max_idx]);
            // v_map_cub_labidx.push_back(std::make_pair(frm_label, cub_idx));
            
            v_frame_centroids.push_back(frame_obj[frm_idx]->v_all_cuboids[0]->centroid);
            v_frame_covs.push_back(frame_obj[frm_idx]->v_all_cuboids[0]->cov);

            v_map_centroids.push_back(map_obj[map_idx]->v_all_cuboids[cub_idx]->centroid);
            v_map_covs.push_back(map_obj[map_idx]->v_all_cuboids[cub_idx]->cov);

            v_weight.push_back(maWeights[frm_label]);
            
            // std::cout << "Check conflicts for pair: " << frm_idx << " - " << map_idx << std::endl;
            for(size_t j=0; j<NUM_PAIR_CANDIDATE; ++j)
            {
                if(frm_idx==v_pair_all[j].first || map_idx==v_pair_all[j].second.first)
                {
                    // conflicts found, set to zero
                    xStar(j) = 0.;
                    // std::cout << "              " << v_pair_all[j].first << " - " 
                    //           << v_pair_all[j].second.first << ", " << v_pair_all[j].second.second << std::endl;
                }
            }
        }
        else 
        {
            break;
        }
    } // -i

    // Use the inliers to get reloc pose candidate
    if(v_inlier_candidates.size() < 3){
        std::cout << " ! No enough inliers. At least 3 needed." << std::endl;
        return pose_candidates;
    }
    else
    {
        // AO
        b_enough_corresp = true;
        Eigen::Matrix4d pose_ao;
        std::vector<bool> outliers;
        float inlier_r, loss;
        b_recovered = PoseEstimator::RANSAC(v_frame_centroids, v_map_centroids,
                                                 v_weight, outliers, pose_ao, 
                                                 inlier_r, loss);        
        // pose_candidates.push_back(pose_ao);

        //- Remove outliers
        int rev_count = 0;
        for(size_t i=0; i<outliers.size(); ++i)
        {
            if(outliers[i]){
                v_map_centroids.erase(v_map_centroids.begin()+i-rev_count);
                v_frame_centroids.erase(v_frame_centroids.begin()+i-rev_count);
                v_map_covs.erase(v_map_covs.begin()+i-rev_count);
                v_frame_covs.erase(v_frame_covs.begin()+i-rev_count);
                rev_count++;

                v_inlier_candidates[i].second.second = -1;
                // v_map_cub_labidx[i].second = -1;
            }
        }

        //- Probabilistic Absolute Orientation (GICP)
        Optimizer::GeneralICP(v_map_centroids, v_frame_centroids, v_map_covs, v_frame_covs, pose_ao);
        pose_candidates.push_back(pose_ao);

        vv_inlier_pairs.push_back(v_inlier_candidates);
        // vv_best_map_cub_labidx.push_back(v_map_cub_labidx);
    }
    
    return pose_candidates;
}

std::vector<Eigen::Matrix4d> Relocalizer::object_relocalization(std::vector<std::shared_ptr<Object3d>> frame_obj, 
                                                 std::vector<std::shared_ptr<Object3d>> map_obj,
                                                 std::vector<std::pair<int, int>>& v_best_map_cub_labidx,
                                                 bool & b_enough_corresp, bool & b_recovered)
{
    // frame_obj and map_obj are stored in matched order before passing here
    // std::cout << frame_obj.size() << " - " << map_obj.size() << std::endl;
    std::vector<Eigen::Matrix4d> v_pose_candidates;

    //- Get object correspondences
    // first is map, second is frame
    std::vector<std::vector<std::shared_ptr<Cuboid3d>>> v_map_object_cuboids;
    std::vector<Eigen::Vector3d> v_frame_centroids, v_map_centroids_tmp;
    std::vector<Eigen::Matrix3d> v_frame_covs, v_map_covs_tmp;
    std::vector<double> v_weight;
    std::vector<int> v_label;
    int num_all_comb = 1;
    // std::cout << "==== Number of cuboids in each matched object: " << std::endl;
    for(size_t i=0; i<map_obj.size(); ++i)
    {
        num_all_comb *= map_obj[i]->v_all_cuboids.size();
        // std::cout << "  Object label " << map_obj[i]->label
        //           << ", number of cuboids is " << map_obj[i]->v_all_cuboids.size()
        //           << std::endl;

        v_label.push_back(map_obj[i]->label);
        v_weight.push_back(maWeights[map_obj[i]->label]);

        v_map_centroids_tmp.push_back(map_obj[i]->pos);
        v_frame_centroids.push_back(frame_obj[i]->pos);

        v_map_covs_tmp.push_back(map_obj[i]->cov);
        v_frame_covs.push_back(frame_obj[i]->v_all_cuboids[0]->cov_propagated);

        v_map_object_cuboids.push_back(map_obj[i]->v_all_cuboids);
    }
    
    if(v_frame_centroids.size() >= 3)
        b_enough_corresp = true;
    else
        return v_pose_candidates;

    //- Get all centroid combinations
    Eigen::Matrix4d pose;
    std::vector<bool> outliers;
    std::vector<Eigen::Vector3d> v_best_map_mean;
    std::vector<Eigen::Matrix3d> v_best_map_cov;
    float inlier_ratio, loss;
    loss = std::numeric_limits<float>::max();

    // std::cout << "==== CHECK Combination selections: " << std::endl;
    for(size_t i=0; i<num_all_comb; ++i)
    {
        std::vector<Eigen::Vector3d> tmp = v_map_centroids_tmp;
        std::vector<Eigen::Matrix3d> tmpC = v_map_covs_tmp;
        std::vector<std::pair<int, int>> tmpLabIdx;
        int pre_comb_num = 1;
        for(size_t j=0; j<tmp.size(); ++j){
            int n = v_map_object_cuboids[j].size();
            // if(n==1)
            //     continue;
            int idx = (i/pre_comb_num) % n;
            tmpLabIdx.push_back(std::make_pair(v_label[j], idx));

            // update correspondence index 
            tmp[j] = v_map_object_cuboids[j][idx]->centroid;
            tmpC[j] = v_map_object_cuboids[j][idx]->cov;

            // std::cout << idx << "(" << n << ")" << v_label[j] << " - "
            //           << tmp[j](0) << ", " << tmp[j](1) << ", " << tmp[j](2) 
            //           << std::endl;

            pre_comb_num *= n;
        }

        // SOLVE ABSOLUTE ORIENTATION //
        Eigen::Matrix4d tmp_pose;
        std::vector<bool> tmp_outliers;
        float tmp_inlier_r, tmp_loss;
        bool tmp_b_recov = PoseEstimator::RANSAC(v_frame_centroids, tmp,
                                                 v_weight, tmp_outliers, tmp_pose, 
                                                 tmp_inlier_r, tmp_loss);
        // bool tmp_b_recov = PoseEstimator::absolute_orientation(v_frame_centroids, tmp, tmp_pose);
        // // evaluate loss
        // tmp_loss = PoseEstimator::compute_residual(v_frame_centroids, tmp, tmp_pose);
        
        if(tmp_loss < loss && tmp_b_recov)
        {
            pose = tmp_pose;
            outliers = tmp_outliers;
            v_best_map_mean = tmp;
            v_best_map_cov = tmpC;
            v_best_map_cub_labidx = tmpLabIdx;

            b_recovered = tmp_b_recov;

            // inlier_ratio = tmp_inlier_r;
            loss = tmp_loss;
        }

        // std::cout << std::endl;
    }
    Eigen::Matrix4d pose_ao = pose;
    v_pose_candidates.push_back(pose_ao);

    //- Remove outliers
    int rev_count = 0;
    for(size_t i=0; i<outliers.size(); ++i)
    {
        // std::cout << v_label[i] << "/" << outliers[i] << " - ";
        if(outliers[i]){
            v_best_map_mean.erase(v_best_map_mean.begin()+i-rev_count);
            v_frame_centroids.erase(v_frame_centroids.begin()+i-rev_count);
            v_best_map_cov.erase(v_best_map_cov.begin()+i-rev_count);
            v_frame_covs.erase(v_frame_covs.begin()+i-rev_count);
            rev_count++;

            v_best_map_cub_labidx[i].second = -1;
        }
    }
    // std::cout << std::endl;

    //- Probabilistic Absolute Orientation (GICP)
    Optimizer::GeneralICP(v_best_map_mean, v_frame_centroids, v_best_map_cov, v_frame_covs, pose);
    v_pose_candidates.push_back(pose);

    return v_pose_candidates;
}

// One instance per object class assumed.
std::vector<Eigen::Matrix4d> Relocalizer::object_guided_relocalization(std::vector<std::shared_ptr<Object3d>> frame_obj,
                                                          std::vector<std::shared_ptr<Object3d>> map_obj,
                                                          std::vector<std::pair<int, int>>& v_best_map_cub_labidx,
                                                          bool & b_enough_corresp, bool & b_recovered)
{
    std::vector<Eigen::Matrix4d> v_pose_candidates;
    //- Get object correspondences
    // first is map, second is frame
    std::vector<std::vector<std::shared_ptr<Cuboid3d>>> v_map_object_cuboids;
    std::vector<Eigen::Vector3d> v_frame_centroids, v_map_centroids_tmp;
    std::vector<Eigen::Matrix3d> v_frame_covs, v_map_covs_tmp;
    std::vector<double> v_weight;
    std::vector<int> v_label;
    int num_all_comb = 1;
    // std::cout << "==== Number of cuboids in each matched object: " << std::endl;
    for(size_t i=0; i<map_obj.size(); ++i)
    {
        for(size_t j=0; j<frame_obj.size(); ++j)
        {
            if(map_obj[i]->label == frame_obj[j]->label)
            {
                num_all_comb *= map_obj[i]->v_all_cuboids.size();
                // std::cout << "  Object label " << map_obj[i]->label
                //           << ", number of cuboids is " << map_obj[i]->v_all_cuboids.size()
                //           << std::endl;

                v_label.push_back(map_obj[i]->label);
                v_weight.push_back(maWeights[map_obj[i]->label]);

                v_map_centroids_tmp.push_back(map_obj[i]->pos);
                v_frame_centroids.push_back(frame_obj[j]->pos);

                v_map_covs_tmp.push_back(map_obj[i]->cov);
                v_frame_covs.push_back(frame_obj[j]->v_all_cuboids[0]->cov_propagated);

                v_map_object_cuboids.push_back(map_obj[i]->v_all_cuboids);
            }
        }
    }

    if(v_frame_centroids.size() >= 3)
        b_enough_corresp = true;
    else
        return v_pose_candidates;

    //- Get all centroid combinations
    Eigen::Matrix4d pose;
    std::vector<bool> outliers;
    std::vector<Eigen::Vector3d> v_best_map_mean;
    std::vector<Eigen::Matrix3d> v_best_map_cov;
    float inlier_ratio, loss;
    loss = std::numeric_limits<float>::max();

    // std::cout << "==== CHECK Combination selections: " << std::endl;
    for(size_t i=0; i<num_all_comb; ++i)
    {
        std::vector<Eigen::Vector3d> tmp = v_map_centroids_tmp;
        std::vector<Eigen::Matrix3d> tmpC = v_map_covs_tmp;
        std::vector<std::pair<int, int>> tmpLabIdx;
        int pre_comb_num = 1;
        for(size_t j=0; j<tmp.size(); ++j){
            int n = v_map_object_cuboids[j].size();
            // if(n==1)
            //     continue;
            int idx = (i/pre_comb_num) % n;
            tmpLabIdx.push_back(std::make_pair(v_label[j], idx));

            // update correspondence index 
            tmp[j] = v_map_object_cuboids[j][idx]->centroid;
            tmpC[j] = v_map_object_cuboids[j][idx]->cov;

            // std::cout << idx << "(" << n << ")" << v_label[j] << " - "
            //           << tmp[j](0) << ", " << tmp[j](1) << ", " << tmp[j](2) 
            //           << std::endl;

            pre_comb_num *= n;
        }

        // SOLVE ABSOLUTE ORIENTATION //
        Eigen::Matrix4d tmp_pose;
        std::vector<bool> tmp_outliers;
        float tmp_inlier_r, tmp_loss;
        bool tmp_b_recov = PoseEstimator::RANSAC(v_frame_centroids, tmp,
                                                 v_weight, tmp_outliers, tmp_pose, 
                                                 tmp_inlier_r, tmp_loss);
        if(tmp_loss < loss && tmp_b_recov)
        {
            pose = tmp_pose;
            outliers = tmp_outliers;
            v_best_map_mean = tmp;
            v_best_map_cov = tmpC;
            v_best_map_cub_labidx = tmpLabIdx;

            b_recovered = tmp_b_recov;

            inlier_ratio = tmp_inlier_r;
            loss = tmp_loss;
        }

        // std::cout << std::endl;
    }
    Eigen::Matrix4d pose_ao = pose;
    v_pose_candidates.push_back(pose_ao);

    //- Remove outliers
    int rev_count = 0;
    for(size_t i=0; i<outliers.size(); ++i)
    {
        // std::cout << v_label[i] << "/" << outliers[i] << " - ";
        if(outliers[i]){
            v_best_map_mean.erase(v_best_map_mean.begin()+i-rev_count);
            v_frame_centroids.erase(v_frame_centroids.begin()+i-rev_count);
            v_best_map_cov.erase(v_best_map_cov.begin()+i-rev_count);
            v_frame_covs.erase(v_frame_covs.begin()+i-rev_count);
            rev_count++;

            v_best_map_cub_labidx[i].second = -1;
        }
    }
    // std::cout << std::endl;

    //- Probabilistic Absolute Orientation (GICP)
    Optimizer::GeneralICP(v_best_map_mean, v_frame_centroids, v_best_map_cov, v_frame_covs, pose);
    v_pose_candidates.push_back(pose);

    return v_pose_candidates;
}

void Relocalizer::set_maWeights(std::map<int, double> new_weights)
{
    maWeights = new_weights;
}

Eigen::Matrix4d Relocalizer::NOCS_relocalization(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                                 std::vector<std::shared_ptr<Cuboid3d>> map_obj,
                                                 bool & b_enough_corresp, bool & b_recovered)
{
    if(frame_obj.size() < 1){
        return Eigen::Matrix4d::Identity();
    } else {
        b_enough_corresp = true;
    }
    // for each object, we have pose in the world coordinate system and camera coordinate system
    // calculate a pose from each pair of object coorespondence
    // (weighted) average over all poses
    std::vector<Eigen::Quaterniond> vRotations;
    std::vector<Eigen::Vector3d> vTranslations;
    std::vector<double> vWeights;
    for(size_t i=0; i<map_obj.size(); ++i)
    {
        int label = map_obj[i]->label;
        // loop through objects in the test frame
        for(size_t j=0; j<frame_obj.size(); ++j)
        {
            if(label != frame_obj[j]->label)
                continue;

            Eigen::Matrix4d tmpPose = map_obj[i]->pose * frame_obj[j]->pose.inverse();
            Eigen::Matrix3d tmpRot = tmpPose.topLeftCorner(3,3);
            Eigen::Quaterniond tmpQua(tmpRot);
            // std::cout << tmpQua.w() << ",\n" << tmpQua.vec() << std::endl;
            Eigen::Vector3d tmpTrans = tmpPose.topRightCorner(3,1);
            vRotations.push_back(tmpQua);
            vTranslations.push_back(tmpTrans);
            vWeights.push_back(1.);

            // std::cout << "Pose from object #" << label << ":  "
            //           << tmpQua.w() << ", " << tmpQua.x() << ", " << tmpQua.y() << ", " << tmpQua.z() << "; " 
            //           << tmpTrans(0) << ", " << tmpTrans(1) << ", " << tmpTrans(2)
            //           << std::endl;
        } // j
    } // i
    // average over all the pose
    Eigen::Matrix4d candidate = pose_average(vRotations, vTranslations, vWeights);
    b_recovered = true;

    return candidate;
}

Eigen::Matrix4d Relocalizer::pose_average(std::vector<Eigen::Quaterniond> vRot, 
                                          std::vector<Eigen::Vector3d> vTrans,
                                          std::vector<double> vW)
{
    Eigen::Matrix4d avgPose = Eigen::Matrix4d::Identity();
    Eigen::Vector3d sumTrans = Eigen::Vector3d::Zero();
    Eigen::Matrix4d sumQQT = Eigen::Matrix4d::Zero();
    double sumW = 0;
    
    for(size_t i=0; i<vRot.size(); ++i)
    {
        // average the translation
        sumTrans += vW[i] * vTrans[i];
        // average the quaternion
        Eigen::Quaterniond tmpQ = vRot[i];
        Eigen::Vector4d tmpVQ(tmpQ.w(), tmpQ.x(), tmpQ.y(), tmpQ.z());
        // tmpVQ << tmpQ.w(), tmpQ.x(), tmpQ.y(), tmpQ.z();
        sumQQT += vW[i] * tmpVQ * tmpVQ.transpose();
        sumW += vW[i];
    }
    sumTrans /= sumW;
    sumQQT /= sumW;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(sumQQT);
    // std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
    // std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
    //     << "corresponding to these eigenvalues:\n"
    //     << eigensolver.eigenvectors() << std::endl;
    // Get the eigenvector corresponding to largest eigen value
    Eigen::Vector4d evQ = eigensolver.eigenvectors().col(3);
    // std::cout << evQ << std::endl;
    Eigen::Quaterniond Q(evQ(0), evQ(1), evQ(2), evQ(3));
    avgPose.topLeftCorner(3, 3) = Q.toRotationMatrix();
    avgPose.topRightCorner(3, 1) = sumTrans;

    // std::cout << "Averaged pose is:     " 
    //           << evQ(0) << ", " << evQ(1) << ", " << evQ(2) << ", " << evQ(3) << "; " 
    //           << sumTrans(0) << ", " << sumTrans(1) << ", " << sumTrans(2)
    //           << std::endl; 
    // std::cout << avgPose << std::endl;

    return avgPose;
}

Eigen::Matrix4d Relocalizer::pose_vec_relocalization(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                                     std::vector<std::shared_ptr<Cuboid3d>> map_obj,
                                                     bool & b_enough_corresp, bool & b_recovered)
{
    if(frame_obj.size() < 1){
        return Eigen::Matrix4d::Identity();
    }
    // for each object, we have pose in the world coordinate system and camera coordinate system
    // calculate a pose from each pair of object coorespondence
    // (weighted) average over all poses
    std::vector<Eigen::Quaterniond> vRotations;
    std::vector<Eigen::Vector3d> vTranslations;
    std::vector<double> vWeight;
    std::vector<Eigen::Vector3d> v_map_vecs, v_frame_vecs;
    std::vector<double> v_w;
    int pose_count = 0,
        vec_count = 0;
    for(size_t i=0; i<map_obj.size(); ++i)
    {
        int label = map_obj[i]->label;
        // loop through objects in the test frame
        bool use_pose, use_vec;
        // use vectors
            // if(std::find(label_pose.begin(), label_pose.end(), label) != label_pose.end()){
            //     use_pose = true;
            //     use_vec = false;
            // } else if (std::find(label_vec.begin(), label_vec.end(), label) != label_vec.end()){
            //     use_pose = false;
            //     use_vec = true;
            // } else {
            //     use_pose = false;
            //     use_vec = false;
            //     std::cout << "Object " << label << " not recognized!!!!" << std::endl;
            //     continue;
            // }
        // use direct labels
        if(label == 3 || label == 5){
            use_pose = true;
            use_vec = false;
        } else if (label==1 || label==2 || label==4 || label==6){
            use_pose = false;
            use_vec = true;
        } else {
            use_pose = false;
            use_vec = false;
            std::cout << "Object " << label << " not recognized!!!!" << std::endl;
            continue;
        }
        for(size_t j=0; j<frame_obj.size(); ++j)
        {
            if(label != frame_obj[j]->label)
                continue;

            if(use_pose){
                Eigen::Matrix4d tmpPose = compute_pose_from_poses(frame_obj[j]->pose, map_obj[i]->pose);
                Eigen::Matrix3d tmpRot = tmpPose.topLeftCorner(3,3);
                Eigen::Quaterniond tmpQua(tmpRot);
                // std::cout << tmpQua.w() << ",\n" << tmpQua.vec() << std::endl;
                Eigen::Vector3d tmpTrans = tmpPose.topRightCorner(3,1);
                vRotations.push_back(tmpQua);
                vTranslations.push_back(tmpTrans);
                if(label == 3){
                    // camera
                    vWeight.push_back(0.7);
                }
                if(label == 5){
                    // laptop
                    vWeight.push_back(1.);
                }
                // std::cout << "Pose from object #" << label << ":  "
                //       << tmpQua.w() << ", " << tmpQua.x() << ", " << tmpQua.y() << ", " << tmpQua.z() << "; " 
                //       << tmpTrans(0) << ", " << tmpTrans(1) << ", " << tmpTrans(2)
                //       << std::endl;
                pose_count++;
            } else if(use_vec){
                Eigen::Vector3d y_top_map, center_map, y_top_frame, center_frame;
                y_top_map << map_obj[i]->axes.at<float>(0, 2), 
                             map_obj[i]->axes.at<float>(1, 2), 
                             map_obj[i]->axes.at<float>(2, 2);
                center_map = map_obj[i]->centroid;
                v_map_vecs.push_back(y_top_map);
                v_map_vecs.push_back(center_map);
                y_top_frame << frame_obj[j]->axes.at<float>(0, 2), 
                               frame_obj[j]->axes.at<float>(1, 2), 
                               frame_obj[j]->axes.at<float>(2, 2);
                center_frame = frame_obj[j]->centroid;
                v_frame_vecs.push_back(y_top_frame);
                v_frame_vecs.push_back(center_frame);
                vec_count++;
            } else {
                std::cout << "SHOULDN'T GET HERE, ERROR!!!!" << std::endl;
            }
        } // j
    } // i
    if(vec_count>2 || pose_count>0) {
        b_enough_corresp = true;
    }
    if(vec_count > 2)
    {
        Eigen::Matrix4d tmpPose;
        bool b_rec = PoseEstimator::absolute_orientation(v_frame_vecs, v_map_vecs, tmpPose);
        Eigen::Matrix3d tmpRot = tmpPose.topLeftCorner(3,3);
        Eigen::Quaterniond tmpQua(tmpRot);
        // std::cout << tmpQua.w() << ",\n" << tmpQua.vec() << std::endl;
        Eigen::Vector3d tmpTrans = tmpPose.topRightCorner(3,1);
        vRotations.push_back(tmpQua);
        vTranslations.push_back(tmpTrans);
        vWeight.push_back(0.7);
        // std::cout << "Pose from y-axes:  "
        //               << tmpQua.w() << ", " << tmpQua.x() << ", " << tmpQua.y() << ", " << tmpQua.z() << "; " 
        //               << tmpTrans(0) << ", " << tmpTrans(1) << ", " << tmpTrans(2)
        //               << std::endl;
    }
    // average over all the pose
    Eigen::Matrix4d candidate = pose_average(vRotations, vTranslations, vWeight);
    b_recovered = true;

    return candidate;
}

Eigen::Matrix4d Relocalizer::compute_pose_from_poses(Eigen::Matrix4d pose_frame, 
                                                     Eigen::Matrix4d pose_map)
{
    return pose_map * pose_frame.inverse();
}

Eigen::Matrix4d Relocalizer::semantic_relocalization(std::vector<std::shared_ptr<Cuboid3d>> frame_obj,
                                                     std::vector<std::shared_ptr<Cuboid3d>> map_obj,
                                                     bool b_use_corners,
                                                     Eigen::Matrix4d gtPose,
                                                     bool & b_enough_corresp, bool & b_recovered)
{
    // GET CORRESPONDENCES //
    // b_enough_corresp = false;
    // b_recovered = false;
    // index: obj idx in map; first: obj idx in frame; second: obj label
    std::vector<std::pair<int, int>> vFrameidxLabelPair;
    // store the correspondences
    std::vector<Eigen::Vector3d> v_map_centroids, v_frame_centroids;
    std::vector<double> v_weight;
    // gicp means and covariances
    std::vector<Eigen::Vector3d> vMapMean, vFrmMean;
    std::vector<Eigen::Matrix3d> vMapCov, vFrmCov;
    // -TEST
    std::vector<std::string> vNames;
    // loop through objects in the map
    for(size_t i=0; i<map_obj.size(); ++i)
    {
        int label = map_obj[i]->label;
        // loop through objects in the test frame
        for(size_t j=0; j<frame_obj.size(); ++j)
        {
            if(label != frame_obj[j]->label)
                continue;

            // first rule out the unreliable detections
            float dim_m1 = map_obj[i]->scale * map_obj[i]->dims[0],
                  dim_m2 = map_obj[i]->scale * map_obj[i]->dims[1],
                  dim_m3 = map_obj[i]->scale * map_obj[i]->dims[2],
                  dim_f1 = frame_obj[j]->scale * frame_obj[j]->dims[0],
                  dim_f2 = frame_obj[j]->scale * frame_obj[j]->dims[1],
                  dim_f3 = frame_obj[j]->scale * frame_obj[j]->dims[2];
            if(dim_f1>2*dim_m1 || dim_f2>2*dim_m2 || dim_f3>2*dim_m3)
                continue;

            vFrameidxLabelPair.push_back(std::make_pair(j, frame_obj[j]->label));
            // std::cout << "label = " << label << " with j=" << j << std::endl;
      
            // store centoid pair
            v_map_centroids.push_back(map_obj[i]->centroid);
            v_frame_centroids.push_back(frame_obj[j]->centroid);
            v_weight.push_back(maWeights[label]);

            // store GICP input
            vMapMean.push_back(map_obj[i]->mean);
            vFrmMean.push_back(frame_obj[j]->centroid);
            vMapCov.push_back(map_obj[i]->cov);
            vFrmCov.push_back(frame_obj[j]->cov_propagated); // This is acutally frame uncertainty
            std::cout << map_obj[i]->label << ":" << std::endl
                      << map_obj[i]->cov << std::endl
                      << frame_obj[j]->cov_propagated << std::endl << std::endl;
            // -TEST
            vNames.push_back(maLableText[label]);
        } // j
    } // i
    int num_matched_obj = v_map_centroids.size();
    if(num_matched_obj >= 3)
        b_enough_corresp = true;

    // CALCULATE THE FINAL POSE //
    std::cout << "Index vs Name:" << std::endl;
    for(size_t i=0; i<vNames.size(); ++i){
        std::cout << i << ": " << vNames[i] << std::endl;
    }
    // directly use all inliers for relocalization
    Eigen::Matrix4d pose;
    // b_recovered = PoseEstimator::absolute_orientation(v_frame_centroids, v_map_centroids, pose);
    std::vector<bool> outliers;
    float inlier_ratio, confidence;
    b_recovered = PoseEstimator::RANSAC(v_frame_centroids, v_map_centroids, v_weight, outliers, pose, inlier_ratio, confidence);

    // GICP optimization
    // std::cout << "Outliser? ";
    int rev_count = 0;
    for(size_t i=0; i<outliers.size(); ++i)
    {
        if(outliers[i]){
            vMapMean.erase(vMapMean.begin()+i-rev_count);
            vFrmMean.erase(vFrmMean.begin()+i-rev_count);
            vMapCov.erase(vMapCov.begin()+i-rev_count);
            vFrmCov.erase(vFrmCov.begin()+i-rev_count);
            rev_count++;
        }
    }
    // std::cout << std::endl;
    
    Eigen::Vector3d gtTrans = gtPose.topRightCorner(3,1);
    Eigen::Matrix3d tmpRot_bu = pose.topLeftCorner(3,3);
    Eigen::Quaterniond tmpQua_bu(tmpRot_bu);
    Eigen::Vector3d tmpTrans_bu = pose.topRightCorner(3,1);
    std::cout << "Pose after ao: " << (gtTrans-tmpTrans_bu).norm() 
            //   << tmpQua_bu.w() << ", " << tmpQua_bu.x() << ", " << tmpQua_bu.y() << ", " << tmpQua_bu.z() << "; " 
            //   << tmpTrans_bu(0) << ", " << tmpTrans_bu(1) << ", " << tmpTrans_bu(2)
              << std::endl;

    // - Probabilistic Absolute Orientation
    Optimizer::GeneralICP(vMapMean, vFrmMean, vMapCov, vFrmCov, pose);
    Eigen::Matrix3d tmpRot = pose.topLeftCorner(3,3);
    Eigen::Quaterniond tmpQua(tmpRot);
    Eigen::Vector3d tmpTrans = pose.topRightCorner(3,1);
    std::cout << "Pose after generalized icp: " << (gtTrans-tmpTrans).norm() 
            //   << tmpQua.w() << ", " << tmpQua.x() << ", " << tmpQua.y() << ", " << tmpQua.z() << "; " 
            //   << tmpTrans(0) << ", " << tmpTrans(1) << ", " << tmpTrans(2)
              << std::endl;

    if(b_recovered)
        return pose;
    else
        return Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d Relocalizer::semantic_reloc_with_dic(std::vector<std::shared_ptr<Cuboid3d>> frame_obj,
                                                     std::map<int, std::vector<std::shared_ptr<Cuboid3d>>> map_obj_dic,
                                                     bool b_use_corners,
                                                     Eigen::Matrix4d gtPose,
                                                     bool & b_enough_corresp, bool & b_recovered)
{
    // int thre_obs = 9;
    int thre_obs = 2; // for short sequences only

    // GET CORRESPONDENCES //
    // store the valid map detections
    std::map<int, std::vector<Eigen::Vector3d>> label_cent_dic;
    std::map<int, std::vector<Eigen::Vector3d>> label_mean_dic;
    std::map<int, std::vector<Eigen::Matrix3d>> label_cov_dic;
    int num_all_comb = 1;
    // store the correspondences - outer for combination and inner for obj pairs
    std::vector<Eigen::Vector3d> v_frame_centroids, v_map_tmp;
    std::vector<double> v_weight;
    std::vector<int> v_label;
    // gicp means and covariances
    std::vector<std::vector<Eigen::Vector3d>> vvMapMean;
    std::vector<std::vector<Eigen::Matrix3d>> vvMapCov;
    std::vector<Eigen::Vector3d> vFrmMean, vMap_tmpM;
    std::vector<Eigen::Matrix3d> vFrmCov, vMap_tmpC;

    // loop the map
    std::map<int, std::vector<std::shared_ptr<Cuboid3d>>>::iterator it;
    for(it=map_obj_dic.begin(); it!=map_obj_dic.end(); ++it){
        int label = it->first;
        // loop through objects in the test frame
        for(size_t j=0; j<frame_obj.size(); ++j)
        {
            if(label != frame_obj[j]->label)
                continue;

            // get map object information
            std::vector<Eigen::Vector3d> v_valid_cent;
            std::vector<Eigen::Vector3d> v_valid_mean;
            std::vector<Eigen::Matrix3d> v_valid_cov;
            for(size_t c=0; c<it->second.size(); ++c)
            {
                if(it->second[c]->observation >= thre_obs){
                    v_valid_cent.push_back(it->second[c]->centroid);
                    v_valid_mean.push_back(it->second[c]->mean);
                    v_valid_cov.push_back(it->second[c]->cov);
                }
            }
            if(!v_valid_cent.empty()){
                label_cent_dic.insert(std::make_pair(label, v_valid_cent));
                label_mean_dic.insert(std::make_pair(label, v_valid_mean));
                label_cov_dic.insert(std::make_pair(label, v_valid_cov));
                num_all_comb *= v_valid_cent.size();
            } else {
                std::cout << "!! Object (label) " << label << " has no valid detection in the map." << std::endl;
                break;
            }

            // store initial correspondence centoid pair
            v_map_tmp.push_back(v_valid_cent[0]);
            v_frame_centroids.push_back(frame_obj[j]->centroid);
            v_weight.push_back(maWeights[label]);
            v_label.push_back(label);
            // store GICP input
            vMap_tmpM.push_back(v_valid_mean[0]);
            vFrmMean.push_back(frame_obj[j]->centroid);
            vMap_tmpC.push_back(v_valid_cov[0]);
            vFrmCov.push_back(frame_obj[j]->cov_propagated); // This is acutally frame uncertainty
        } // j
    } // it
    
    if(v_frame_centroids.size() >= 3)
        b_enough_corresp = true;
    else
        return Eigen::Matrix4d::Identity();

    // std::cout << "All combination = " << num_all_comb << std::endl;
    // std::map<int, std::vector<Eigen::Vector3d>>::iterator it2;
    // for(it2=label_cent_dic.begin(); it2!=label_cent_dic.end(); ++it2){
    //     std::cout << it2->second.size() << " - ";
    // }
    // std::cout << std::endl;

    // get all combination
    Eigen::Matrix4d pose;
    std::vector<bool> outliers;
    float inlier_ratio, loss;
    loss = std::numeric_limits<float>::max();
    int best_comb_idx = -1;
    for(size_t i=0; i<num_all_comb; ++i)
    {
        std::vector<Eigen::Vector3d> tmp = v_map_tmp;
        std::vector<Eigen::Vector3d> tmpM = vMap_tmpM;
        std::vector<Eigen::Matrix3d> tmpC = vMap_tmpC;
        int pre_comb_num = 1;
        for(size_t j=0; j<tmp.size(); ++j){
            int l = v_label[j];
            int n = label_cent_dic[l].size();
            if(n==1)
                continue;
            int idx = (i/pre_comb_num) % n;
            // std::cout << idx << " - ";

            // update correspondence index 
            tmp[j] = label_cent_dic[l][idx];
            tmpM[j] = label_mean_dic[l][idx];
            tmpC[j] = label_cov_dic[l][idx];

            pre_comb_num *= n;
        }
        // std::cout << std::endl;
        // store this set of correspondence combination
        // vv_map_centroids.push_back(tmp);
        vvMapMean.push_back(tmpM);
        vvMapCov.push_back(tmpC);

    // SOLVE ABSOLUTE ORIENTATION //
        Eigen::Matrix4d tmp_pose;
        std::vector<bool> tmp_outliers;
        float tmp_inlier_r, tmp_loss;
        bool tmp_b_recov = PoseEstimator::RANSAC(v_frame_centroids, tmp,
                                                 v_weight, tmp_outliers, tmp_pose, 
                                                 tmp_inlier_r, tmp_loss);
        if(tmp_loss < loss && tmp_b_recov)
        {
            pose = tmp_pose;
            outliers = tmp_outliers;
            inlier_ratio = tmp_inlier_r;
            b_recovered = tmp_b_recov;
            loss = tmp_loss;
            best_comb_idx = i;
        }
    }

    // CALCULATE THE FINAL POSE //
    std::vector<Eigen::Vector3d> vMapMean = vvMapMean[best_comb_idx];
    std::vector<Eigen::Matrix3d> vMapCov = vvMapCov[best_comb_idx];
    int rev_count = 0;
    for(size_t i=0; i<outliers.size(); ++i)
    {
        if(outliers[i]){
            vMapMean.erase(vMapMean.begin()+i-rev_count);
            vFrmMean.erase(vFrmMean.begin()+i-rev_count);
            vMapCov.erase(vMapCov.begin()+i-rev_count);
            vFrmCov.erase(vFrmCov.begin()+i-rev_count);
            rev_count++;
        }
    }
    
    Eigen::Vector3d gtTrans = gtPose.topRightCorner(3,1);
    Eigen::Matrix3d tmpRot_bu = pose.topLeftCorner(3,3);
    Eigen::Quaterniond tmpQua_bu(tmpRot_bu);
    Eigen::Vector3d tmpTrans_bu = pose.topRightCorner(3,1);
    std::cout << "Trans Diff after ao: " << (gtTrans-tmpTrans_bu).norm() 
            //   << tmpQua_bu.w() << ", " << tmpQua_bu.x() << ", " << tmpQua_bu.y() << ", " << tmpQua_bu.z() << "; " 
            //   << tmpTrans_bu(0) << ", " << tmpTrans_bu(1) << ", " << tmpTrans_bu(2)
              << std::endl;

    // Probabilistic Absolute Orientation (GICP)
    Optimizer::GeneralICP(vMapMean, vFrmMean, vMapCov, vFrmCov, pose);
    Eigen::Matrix3d tmpRot = pose.topLeftCorner(3,3);
    Eigen::Quaterniond tmpQua(tmpRot);
    Eigen::Vector3d tmpTrans = pose.topRightCorner(3,1);
    std::cout << "Trans Diff after generalized icp: " << (gtTrans-tmpTrans).norm() 
            //   << tmpQua.w() << ", " << tmpQua.x() << ", " << tmpQua.y() << ", " << tmpQua.z() << "; " 
            //   << tmpTrans(0) << ", " << tmpTrans(1) << ", " << tmpTrans(2)
              << std::endl;
    std::cout << pose << std::endl;

    if(b_recovered)
        return pose;
    else
        return Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d Relocalizer::semantic_reloc_recog(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                         std::vector<std::vector<std::shared_ptr<Cuboid3d>>> maps,
                                         bool b_use_corners,
                                         bool & b_enough_corresp, bool & b_recovered, int & mapId)
{
    float best_loss = std::numeric_limits<float>::max();
    int best_map = 0;
    Eigen::Matrix4d best_pose = Eigen::Matrix4d::Identity();
    
    for(size_t map_idx=0; map_idx<maps.size(); ++map_idx)
    {
        // GET MAP //
        std::vector<std::shared_ptr<Cuboid3d>> map_obj = maps[map_idx];
        std::cout << "Comparing with map#" << map_idx+1 << std::endl;

        // GET CORRESPONDENCES //
        // index: obj idx in map; first: obj idx in frame; second: obj label
        std::vector<std::pair<int, int>> vFrameidxLabelPair;
        // store the correspondences
        std::vector<Eigen::Vector3d> v_map_centroids, v_frame_centroids;
        std::vector<double> v_weight;
        // -TEST
        std::vector<std::string> vNames;
        // loop through objects in the map
        for(size_t i=0; i<map_obj.size(); ++i)
        {
            int label = map_obj[i]->label;
            // loop through objects in the test frame
            for(size_t j=0; j<frame_obj.size(); ++j)
            {
                if(label != frame_obj[j]->label)
                    continue;

                // first rule out the unreliable detections
                float dim_m1 = map_obj[i]->scale * map_obj[i]->dims[0],
                    dim_m2 = map_obj[i]->scale * map_obj[i]->dims[1],
                    dim_m3 = map_obj[i]->scale * map_obj[i]->dims[2],
                    dim_f1 = frame_obj[j]->scale * frame_obj[j]->dims[0],
                    dim_f2 = frame_obj[j]->scale * frame_obj[j]->dims[1],
                    dim_f3 = frame_obj[j]->scale * frame_obj[j]->dims[2];
                if(dim_f1>2*dim_m1 || dim_f2>2*dim_m2 || dim_f3>2*dim_m3)
                    continue;

                vFrameidxLabelPair.push_back(std::make_pair(j, frame_obj[j]->label));
                // std::cout << "label = " << label << " with j=" << j << std::endl;
        
                // store centoid pair
                v_map_centroids.push_back(map_obj[i]->centroid);
                v_frame_centroids.push_back(frame_obj[j]->centroid);
                v_weight.push_back(maWeights[label]);

                // -TEST
                vNames.push_back(maLableText[label]);
            } // j
        } // i
        int num_matched_obj = v_map_centroids.size();
        if(num_matched_obj >= 3)
            b_enough_corresp = true;

        // CALCULATE THE FINAL POSE //
        std::cout << "Index vs Name:" << std::endl;
        for(size_t i=0; i<vNames.size(); ++i){
            std::cout << i << ": " << vNames[i] << std::endl;
        }
        // directly use all inliers for relocalization
        Eigen::Matrix4d pose;
        std::vector<bool> outliers;
        float inlier_ratio, loss;
        bool tmp_b_recovered = PoseEstimator::RANSAC(v_frame_centroids, v_map_centroids, v_weight, outliers, pose, inlier_ratio, loss);

        std::cout << "Map id: " << map_idx+1 << " with loss: " << loss << " - " << best_loss << std::endl;
        if(loss < best_loss){
            best_loss = loss;
            best_map = map_idx+1;
            best_pose = pose;
            b_recovered = tmp_b_recovered;
        }
    }

    if(b_recovered){
        mapId = best_map;
        return best_pose;
    }else{
        return Eigen::Matrix4d::Identity();
    }
}
    

} // namespace fusion
