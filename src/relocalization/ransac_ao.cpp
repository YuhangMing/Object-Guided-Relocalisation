#include "relocalization/ransac_ao.h"

#define MAX_RANSAC_ITER 200

namespace fusion
{

bool PoseEstimator::absolute_orientation(
    std::vector<Eigen::Vector3d> src,
    std::vector<Eigen::Vector3d> dst,
    std::vector<bool> outliers,
    Eigen::Matrix4d &estimate)
{
    // src: Global coordinate center; dst: Frame coordinate center
    //! Must initialize before using
    //! Eigen defaults to random numbers
    estimate = Eigen::Matrix4d::Identity();
    Eigen::Vector3d src_pts_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d dst_pts_sum = Eigen::Vector3d::Zero();
    int no_inliers = 0;

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            // no_inliers += weight[i];
            no_inliers++;
            src_pts_sum += src[i];
            dst_pts_sum += dst[i];
            M += src[i] * dst[i].transpose();
        }
    }

    //! Compute centroids
    src_pts_sum /= no_inliers;
    dst_pts_sum /= no_inliers;
    M -= no_inliers * (src_pts_sum * dst_pts_sum.transpose());

    // double Sxx=0, Sxy=0, Sxz=0, 
        //        Syx=0, Syy=0, Syz=0,
        //        Szx=0, Szy=0, Szz=0;
        // for (int i = 0; i < src.size(); ++i)
        // {
        //     if (!outliers[i])
        //     {
        //         // Horn 1987
        //         Sxx += (src[i](0) - src_pts_sum(0)) * (dst[i](0) - dst_pts_sum(0));
        //         Sxy += (src[i](0) - src_pts_sum(0)) * (dst[i](1) - dst_pts_sum(1));
        //         Sxz += (src[i](0) - src_pts_sum(0)) * (dst[i](2) - dst_pts_sum(2));
        //         Syx += (src[i](1) - src_pts_sum(1)) * (dst[i](0) - dst_pts_sum(0));
        //         Syy += (src[i](1) - src_pts_sum(1)) * (dst[i](1) - dst_pts_sum(1));
        //         Syz += (src[i](1) - src_pts_sum(1)) * (dst[i](2) - dst_pts_sum(2));
        //         Szx += (src[i](2) - src_pts_sum(2)) * (dst[i](0) - dst_pts_sum(0));
        //         Szy += (src[i](2) - src_pts_sum(2)) * (dst[i](1) - dst_pts_sum(1));
        //         Szz += (src[i](2) - src_pts_sum(2)) * (dst[i](2) - dst_pts_sum(2));
        //     }
    // }

    const auto svd = M.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto MatU = svd.matrixU();
    // const auto u1 = MatU.topLeftCorner(3, 1);
    // const auto u2 = MatU.topLeftCorner(3, 2).topRightCorner(3, 1);
    // const auto u3 = MatU.topRightCorner(3, 1);
    const auto MatV = svd.matrixV();
    // const auto vals = svd.singularValues();
    Eigen::Matrix3d R;

    // Arun et al. 1987, use simple svd
    R = MatV * MatU.transpose();
    if(R.determinant() < 0)
    {
        // std::cout << "Validating Rotation Matrx: " << R.determinant() << std::endl;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        I(2, 2) = -1;
        // I(2, 2) = (MatU * MatV.transpose()).determinant();
        R = MatV * I * MatU.transpose();
        // R = MatU * I * MatV.transpose();
    }
    // std::cout << R << std::endl;

    // // Horn et al. 1988
        // // std::cout << MatU << std::endl;
        // // std::cout << u1 << std::endl;
        // // std::cout << u2 << std::endl;
        // // std::cout << u3 << std::endl;
        // std::cout << vals << std::endl;
        // if(vals(2) == 0){
        //     R = M * (u1*u1.transpose()/vals(0) + u2*u2.transpose()/vals(1)) + u3*u3.transpose();
        //     std::cout << R << std::endl;
        //     R = M * (u1*u1.transpose()/vals(0) + u2*u2.transpose()/vals(1)) - u3*u3.transpose();
        // } else {
        //     R = M * (u1*u1.transpose()/vals(0) + u2*u2.transpose()/vals(1) + u3*u3.transpose()/vals(2));
        // }
    // std::cout << R << std::endl;

    // // Horn et al. 1987
        // Eigen::Matrix4d P;
        // P <<Sxx+Syy+Szz, Syz-Szy, Szx-Sxz, Sxy-Syx,
        //     Syz-Szy, Sxx-Syy-Szz, Sxy+Syx, Szx+Sxz,
        //     Szx-Sxz, Sxy+Syz, Syy-Sxx-Szz, Syz+Szy,
        //     Sxy-Syx, Szx+Sxz, Syz+Szy, Szz-Sxx-Syy;
        // Eigen::EigenSolver<Eigen::Matrix4d> es(P);
        // const auto eVals = es.eigenvalues();
        // double max_eVal = 0;
        // Eigen::Vector4d q;
        // const auto eVecs = es.eigenvectors();
        // // std::cout << eVals << std::endl;
        // // std::cout << eVecs << std::endl;
        // for(size_t i=0; i<4; ++i)
        // {
        //     if(eVals(i).imag() == 0){
        //         if(eVals(i).real() > max_eVal){
        //             q(0) = eVecs.col(i)(0).real();
        //             q(1) = eVecs.col(i)(1).real();
        //             q(2) = eVecs.col(i)(2).real();
        //             q(3) = eVecs.col(i)(3).real();
        //         }
        //     }
        // }
        // // auto q = es.eigenvectors().col(0);
        // Eigen::Matrix3d R_hat;
        // R_hat << q(0)*q(0)+q(1)*q(1)-q(2)*q(2)-q(3)*q(3), 2*(q(1)*q(2)-q(0)*q(3)), 2*(q(1)*q(3)+q(0)*q(2)),
        //          2*(q(2)*q(1)+q(0)*q(3)), q(0)*q(0)-q(1)*q(1)+q(2)*q(2)-q(3)*q(3), 2*(q(2)*q(3)-q(0)*q(1)),
        //          2*(q(3)*q(1)-q(0)*q(2)), 2*(q(3)*q(2)+q(0)*q(1)), q(0)*q(0)-q(1)*q(1)-q(2)*q(2)+q(3)*q(3);
        // std::cout << R_hat << std::endl;
    // R = R_hat;

    // std::cout << R.determinant() << std::endl;

    const auto t = dst_pts_sum - R * src_pts_sum;

    estimate.topLeftCorner(3, 3) = R;
    estimate.topRightCorner(3, 1) = t;

    return true;
}

bool PoseEstimator::absolute_orientation(
    std::vector<Eigen::Vector3d> src,
    std::vector<Eigen::Vector3d> dst,
    Eigen::Matrix4d &estimate)
{
    // if(weight.empty()){
    //     std::vector<bool> weight(src.size());
    //     std::fill(weight.begin(), weight.end(), 1.);    
    // }
    std::vector<bool> outliers(src.size());
    std::fill(outliers.begin(), outliers.end(), false);
    return absolute_orientation(src, dst, outliers, estimate);
}

int PoseEstimator::evaluate_inliers(
    const std::vector<Eigen::Vector3d> &src,
    const std::vector<Eigen::Vector3d> &dst,
    std::vector<double> weight,
    std::vector<bool> &outliers,
    const Eigen::Matrix4d &estimate,
    double &loss)
{
    int no_inliers = 0;
    loss = 0;
    double dist_thresh = 0.1;
    const auto &R = estimate.topLeftCorner(3, 3);
    const auto &t = estimate.topRightCorner(3, 1);

    std::fill(outliers.begin(), outliers.end(), true);
    for (int i = 0; i < src.size(); ++i)
    {
        double dist = (dst[i] - (R * src[i] + t)).norm() * weight[i];
        // std::cout << dst[i](0) << ", " << dst[i](1) << ", " << dst[i](2) << " - "
        //           << src[i](0) << ", " << src[i](1) << ", " << src[i](2) << " - "
        //           << weight[i] << std::endl;
        
        if(std::isnan(dist)){
            std::cout << " !! Inside RANSAC, nan appears in evaluating inliers. CHECK it!!" << std::endl;
            continue;
        }
        if (dist <= dist_thresh)
        {
            no_inliers++;
            // loss += 1;
            outliers[i] = false;
            // std::cout << " - $$ " << i << " dist=" << dist << std::endl;
        } else {
            loss += dist;
            // std::cout << " - !! " << i << " dist=" << dist << std::endl;
        }
    }

    return no_inliers;
}

float PoseEstimator::compute_residual(
    const std::vector<Eigen::Vector3d> &src,
    const std::vector<Eigen::Vector3d> &dst,
    const Eigen::Matrix4d &pose_estimate)
{
    double residual_sum = 0;
    const auto &R = pose_estimate.topLeftCorner(3, 3);
    const auto &t = pose_estimate.topRightCorner(3, 1);

    for (int i = 0; i < src.size(); ++i)
    {
        residual_sum += (dst[i] - (R * src[i] + t)).norm();
    }

    return float(residual_sum);
}

bool PoseEstimator::RANSAC(
    const std::vector<Eigen::Vector3d> &src,
    const std::vector<Eigen::Vector3d> &dst,
    std::vector<double> &weight,
    std::vector<bool> &outliers,
    Eigen::Matrix4d &estimate,
    float &inlier_ratio, float &output_loss)
{
    const auto size = src.size();
    inlier_ratio = 0.f;
    int no_iter = 0;
    output_loss = 0.f;
    int best_no_inlier = 0;
    double best_loss = std::numeric_limits<double>::max();
    bool best_valid = false;

    //! Compute pose estimate
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    if (outliers.size() != size)
        outliers.resize(size);
    std::vector<bool> tmp_outlier = outliers;

    // loop through all possibilities
    for(size_t i=0; i<size; ++i){
        for(size_t j=i; j<size; ++j){
            if(j == i)
                continue;
            for(size_t k=j; k<size; ++k){
                if(k == i || k == j)
                    continue;
                //! combination idx
                // std::cout << "combination: " << i << "-" << j << "-" << k << ": " << std::endl;
                //! Src points corresponds to the frame
                std::vector<Eigen::Vector3d> src_pts = {src[i], src[j], src[k]};
                //! Dst points correspond to the map
                std::vector<Eigen::Vector3d> dst_pts = {dst[i], dst[j], dst[k]};
                // //! Corresponding weights
                // std::vector<double> cor_weights = {weight[i], weight[j], weight[k]};

                //! Check if the 3 points are co-linear
                double src_d = (src_pts[1] - src_pts[0]).cross(src_pts[2] - src_pts[0]).norm();
                double dst_d = (dst_pts[1] - dst_pts[0]).cross(dst_pts[2] - dst_pts[0]).norm();
                // std::cout << " - Check co-linear..." << std::endl;
                if (src_d < 1e-6 || dst_d < 1e-6)
                    continue;

                // ! Compute candiate pose
                const auto valid = absolute_orientation(src_pts, dst_pts, pose);
                // std::cout << " - Compute pose... " << valid << std::endl;

                if (valid)
                {
                    //! Check for outliers
                    double loss;
                    const auto no_inliers = evaluate_inliers(src, dst, weight, tmp_outlier, pose, loss);
                    // std::cout << " - Evaluating model: " << no_inliers 
                    //           << " - " << loss << std::endl;

                    // GOAL: want to keep inlier as many as possible.
                    // if # of inliers are same, use sum dist as loss function
                    // std::cout << "Inlier: " << no_inliers << "/" << best_no_inlier 
                    //           << "; loss: " << loss << "/" << best_loss << std::endl;
                    if(no_inliers > best_no_inlier)
                    {
                        // std::cout << "Use no of inliers " << std::endl;
                        best_no_inlier = no_inliers;
                        best_loss = loss;
                        inlier_ratio = (float)no_inliers / src.size();
                        float confidence = 1 - pow((1 - pow(inlier_ratio, 3)), no_iter + 1);
                        
                        outliers = tmp_outlier;
                        estimate = pose;
                        best_valid = valid;
                    } 
                    else if(no_inliers == best_no_inlier)
                    {
                        // std::cout << "Compare loss " << std::endl;
                        if (loss < best_loss)
                        {
                            // std::cout << "-- get better loss" << std::endl;
                            best_no_inlier = no_inliers;
                            best_loss = loss;
                            inlier_ratio = (float)no_inliers / src.size();
                            float confidence = 1 - pow((1 - pow(inlier_ratio, 3)), no_iter + 1);
                            outliers = tmp_outlier;
                            estimate = pose;
                            best_valid = valid;
                        }
                    }
                    else 
                    {
                        // std::cout << "Discard this combination" << std::endl;
                        // do nothing, discard this combination
                    }
                }
            }// -k
        }// -j
    }// -i

    // // std::cout << "--- GET FINAL INLIERS ---" << std::endl;
    // double final_loss;
    // const auto no_inliers = evaluate_inliers(src, dst, weight, outliers, estimate, final_loss);
    // const auto valid = absolute_orientation(src, dst, outliers, estimate);
    output_loss = best_loss;

    // std::cout << "Final: inlier: " << best_no_inlier 
    //           << "; loss: " << best_loss << std::endl << std::endl;


    return best_valid;

    // std::cout << "Residual: " << residual_sum << std::endl;

    if (!best_valid)
    {
        estimate.setIdentity();
    }
}

}; // namespace fusion