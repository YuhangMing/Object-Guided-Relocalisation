#include "detection/map_cuboid.h"
#include <ctime>

namespace fusion
{

Cuboid3d::Cuboid3d(int l, float c)
{
    label = l;
    confidence = c;
    observation = 1;
}
Cuboid3d::Cuboid3d(int l, float c, Eigen::Matrix4d Rt, std::vector<float> d, float s, Eigen::Matrix3d Sigma_t)
{
    label = l;
    confidence = c;
    observation = 1;
    // network only returns confidence score for the l-object
    // assume the probability for the other classes are the same
    float other_conf = (1-c)/6;
    for(size_t i=0; i<7; ++i)
    {
        if(i == l)
            all_class_confidence.push_back(confidence);
        else
            all_class_confidence.push_back(other_conf);

        // std::cout << all_class_confidence[i] << std::endl;
    }

    // All coordinates are in World Coordinate System
    pose = Rt;
    centroid = pose.topRightCorner(3,1);
    dims = d;
    scale = s;
    // calculate corner_pts and axes of the cuboid
    find_bounding_cuboid();

    // initialize probability paprameters
    mean = centroid;
    cov = Eigen::Matrix3d::Identity(); 
    // initialised with scale factor of 0.003.
    cov = cov * 0.001;
    // // // initialised with half the size of its dimension
    // double cov_scale = 100.;
    // cov(0,0) = s*dims[0]/cov_scale;
    // cov(1,1) = s*dims[1]/cov_scale;
    // cov(2,2) = s*dims[2]/cov_scale;
    vCentroids.push_back(centroid);
    cov_propagated = Sigma_t;

    vScales.push_back(double(scale));
    sigma_scale = 0.001;
}
Cuboid3d::Cuboid3d(std::shared_ptr<Cuboid3d> ref_cub)
{
    label = ref_cub->label;
    confidence = ref_cub->confidence;
    all_class_confidence = ref_cub->all_class_confidence;
    observation = ref_cub->observation;
    v_data_pts = ref_cub->v_data_pts;

    pose = ref_cub->pose;
    dims = ref_cub->dims;
    scale = ref_cub->scale;

    ref_cub->cuboid_corner_pts.copyTo(cuboid_corner_pts);
    ref_cub->axes.copyTo(axes);
    centroid = ref_cub->centroid;
    
    mean = ref_cub->mean;
    cov = ref_cub->cov;
    vCentroids = ref_cub->vCentroids;
    cov_propagated = ref_cub->cov_propagated;

    vScales = ref_cub->vScales;
    sigma_scale = ref_cub->sigma_scale;
}

void Cuboid3d::copyFrom(std::shared_ptr<Cuboid3d> ref_cub)
{
    label = ref_cub->label;
    confidence = ref_cub->confidence;
    all_class_confidence = ref_cub->all_class_confidence;
    observation = ref_cub->observation;
    v_data_pts = ref_cub->v_data_pts;

    pose = ref_cub->pose;
    dims = ref_cub->dims;
    scale = ref_cub->scale;

    ref_cub->cuboid_corner_pts.copyTo(cuboid_corner_pts);
    ref_cub->axes.copyTo(axes);
    centroid = ref_cub->centroid;

    mean = ref_cub->mean;
    cov = ref_cub->cov;
    vCentroids = ref_cub->vCentroids;
    cov_propagated = ref_cub->cov_propagated;

    vScales = ref_cub->vScales;
    sigma_scale = ref_cub->sigma_scale;
}

int Cuboid3d::find_axis_correspondence(std::vector<Eigen::Vector3f> plane_normals, int &sign)
{
    if (axes.empty())
        return -1;
    Eigen::Vector3f cur_y_top;
    cur_y_top << axes.at<float>(0, 2), axes.at<float>(1, 2), axes.at<float>(2, 2);
    Eigen::Vector3f cur_y = cur_y_top - centroid.cast<float>();

    // find correspondence
    // compute the distance, merge the closest ones together
    int bestIdx;
    float bestDist = 0;
    for (size_t j = 0; j < plane_normals.size(); ++j)
    {
        float dist = plane_normals[j].dot(cur_y);
        float tmpSign = 1;
        if (dist < 0)
        {
            tmpSign = -1;
            dist = tmpSign * dist;
        }
        if (dist > bestDist)
        {
            bestDist = dist;
            bestIdx = j;
            sign = tmpSign;
        }
    } //-j

    return bestIdx;
}

void Cuboid3d::align_with_palne_normal(Eigen::Vector3f tar_y)
{
    Eigen::Vector3f cur_y_top;
    cur_y_top << axes.at<float>(0, 2), axes.at<float>(1, 2), axes.at<float>(2, 2);
    Eigen::Vector3f cur_y = cur_y_top - centroid.cast<float>();

    // align
    Eigen::Matrix3f rot = calculate_rotation(cur_y / cur_y.norm(), tar_y / tar_y.norm());
    // update pose
    Eigen::Matrix3d prev_rot = pose.topLeftCorner(3, 3);
    pose.topLeftCorner(3, 3) = rot.cast<double>() * pose.topLeftCorner(3, 3);
    // return calculate_rotation(cur_y/cur_y.norm(), tar_y/tar_y.norm());
}

void Cuboid3d::find_bounding_cuboid()
{
    // find bounding cuboid's corners and axes
    float max_r = scale * dims[0],
          max_g = scale * dims[1],
          max_b = scale * dims[2];
    Eigen::Matrix<float, 3, 4> xyz_axis; // in the order of origin, z, y, x; y-axis is top-down direction
    xyz_axis << 0, 0, 0, max_r / 2,
                0, 0, max_g / 2, 0,
                0, max_b / 2, 0, 0;
    Eigen::Matrix<float, 3, 8> bbox_3d;
    bbox_3d << max_r / 2, max_r / 2, -max_r / 2, -max_r / 2, max_r / 2, max_r / 2, -max_r / 2, -max_r / 2,
               max_g / 2, max_g / 2, max_g / 2, max_g / 2, -max_g / 2, -max_g / 2, -max_g / 2, -max_g / 2,
               max_b / 2, -max_b / 2, max_b / 2, -max_b / 2, max_b / 2, -max_b / 2, max_b / 2, -max_b / 2;

    // find cuboid corners and axes in world coordinate system
    cv::Mat corners(3, 8, CV_32FC1),
            axis(3, 4, CV_32FC1);
    Eigen::Matrix3f Rwn = pose.cast<float>().topLeftCorner(3, 3);
    Eigen::Vector3f twn = pose.cast<float>().topRightCorner(3, 1);
    for (size_t c = 0; c < 8; ++c)
    {
        if (c < 4)
        {
            // Normalized Coordinate -> Camera Coordinate -> World Coordinate
            Eigen::Vector3f ax = Rwn * xyz_axis.col(c) + twn;
            for (size_t r = 0; r < 3; r++)
            {
                axis.at<float>(r, c) = ax(r);
            }
        }
        Eigen::Vector3f cor = Rwn * bbox_3d.col(c) + twn;
        for (size_t r = 0; r < 3; r++)
        {
            corners.at<float>(r, c) = cor(r);
        }
    }
    cuboid_corner_pts = corners.clone();
    axes = axis.clone();
}

void Cuboid3d::add_points(std::vector<Eigen::Vector3f> pts)
{
    // if(v_data_pts.size() <= 333333){
    v_data_pts.assign(pts.begin(), pts.end());
    // } else {
    //     int start = rand() % (333333 - pts.size());
    //     v_data_pts.

    // }
    // cv::Mat cvMat(1, 3, CV_32F);
    // cvMat.at<float>(0,0) = point(0);
    // cvMat.at<float>(0,1) = point(1);
    // cvMat.at<float>(0,2) = point(2);
    // data_pts.push_back(cvMat);
}

Eigen::Matrix3f Cuboid3d::calculate_rotation(Eigen::Vector3f current, Eigen::Vector3f target)
{
    Eigen::Vector3f v = current.cross(target);
    float sin = v.norm();
    float cos = current.dot(target);
    Eigen::Matrix3f Rot, vx;
    vx << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    Rot = Eigen::Matrix3f::Identity() + vx + vx * vx * (1 - cos) / (sin * sin);

    // // Enforce orthogonal requirement
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rot, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // // std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
    // // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
    // // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    // Rot = svd.matrixU() * svd.matrixV();

    return Rot;
}

void Cuboid3d::update_confidence(std::vector<float> obs_confidence)
{
    float normalize_term = 0.;
    for(size_t i=0; i<7; ++i)
    {
      all_class_confidence[i] *= obs_confidence[i];
      normalize_term += all_class_confidence[i];
    }
    for(size_t i=0; i<7; ++i){
      all_class_confidence[i] /= normalize_term;
    //   std::cout << all_class_confidence[i] << std::endl;
    }
    confidence = all_class_confidence[label];
}


Cuboid3d::Cuboid3d(std::string file_name, int start_line)
{
    readFromFile(file_name, start_line);
}
void Cuboid3d::writeToFile(std::string file_name)
{
    std::ofstream semanticfile;
    semanticfile.open(file_name+"-semantic.txt", std::ios::app);
    if (semanticfile.is_open())
    {
        // semanticfile << "New Cuboid:\n";
        semanticfile << label << "," << confidence << "," << observation << "\n";
        semanticfile << pose(0,0) << "," << pose(0,1) << "," << pose(0,2) << "," << pose(0,3) << ","
                    << pose(1,0) << "," << pose(1,1) << "," << pose(1,2) << "," << pose(1,3) << ","
                    << pose(2,0) << "," << pose(2,1) << "," << pose(2,2) << "," << pose(2,3) << ","
                    << pose(3,0) << "," << pose(3,1) << "," << pose(3,2) << "," << pose(3,3) << "\n";
        semanticfile << dims[0] << "," << dims[1] << "," << dims[2] << ", " << scale << ", " << sigma_scale << "\n";
        semanticfile << cuboid_corner_pts.at<float>(0,0) << "," << cuboid_corner_pts.at<float>(0,1) << "," << cuboid_corner_pts.at<float>(0,2) << "," << cuboid_corner_pts.at<float>(0,3) << ","
                    << cuboid_corner_pts.at<float>(0,4) << "," << cuboid_corner_pts.at<float>(0,5) << "," << cuboid_corner_pts.at<float>(0,6) << "," << cuboid_corner_pts.at<float>(0,7) << ","
                    << cuboid_corner_pts.at<float>(1,0) << "," << cuboid_corner_pts.at<float>(1,1) << "," << cuboid_corner_pts.at<float>(1,2) << "," << cuboid_corner_pts.at<float>(1,3) << ","
                    << cuboid_corner_pts.at<float>(1,4) << "," << cuboid_corner_pts.at<float>(1,5) << "," << cuboid_corner_pts.at<float>(1,6) << "," << cuboid_corner_pts.at<float>(1,7) << ","
                    << cuboid_corner_pts.at<float>(2,0) << "," << cuboid_corner_pts.at<float>(2,1) << "," << cuboid_corner_pts.at<float>(2,2) << "," << cuboid_corner_pts.at<float>(2,3) << ","
                    << cuboid_corner_pts.at<float>(2,4) << "," << cuboid_corner_pts.at<float>(2,5) << "," << cuboid_corner_pts.at<float>(2,6) << "," << cuboid_corner_pts.at<float>(2,7) << "\n";
        semanticfile << axes.at<float>(0,0) << ", " << axes.at<float>(0,1) << "," << axes.at<float>(0,2) << "," << axes.at<float>(0,3) << ","
                    << axes.at<float>(1,0) << ", " << axes.at<float>(1,1) << "," << axes.at<float>(1,2) << "," << axes.at<float>(1,3) << ","
                    << axes.at<float>(2,0) << ", " << axes.at<float>(2,1) << "," << axes.at<float>(2,2) << "," << axes.at<float>(2,3) << "\n";
        semanticfile << centroid(0) << "," << centroid(1) << "," << centroid(2) << "\n";
        for(size_t i=0; i<7; ++i){
            float one_score = (all_class_confidence[i] < 0.00000001) ? 0 : all_class_confidence[i];
            if(i < 6)
                semanticfile << one_score << ",";
            else
                semanticfile << one_score << "\n";
        }
        // mean = centroid
        semanticfile << cov(0,0) << "," << cov(0,1) << "," << cov(0,2) << ","
                     << cov(1,0) << "," << cov(1,1) << "," << cov(1,2) << ","
                     << cov(2,0) << "," << cov(2,1) << "," << cov(2,2) << "\n";
        semanticfile << cov_propagated(0,0) << "," << cov_propagated(0,1) << "," << cov_propagated(0,2) << ","
                     << cov_propagated(1,0) << "," << cov_propagated(1,1) << "," << cov_propagated(1,2) << ","
                     << cov_propagated(2,0) << "," << cov_propagated(2,1) << "," << cov_propagated(2,2) << "\n";
        // for(size_t i=0; i<vCentroids.size(); ++i)
        // {
        //     semanticfile << vCentroids[i](0) << "," << vCentroids[i](1) << "," << vCentroids[i](2) << "\n";
        // }
    }
    semanticfile.close();
}
void Cuboid3d::readFromFile(std::string file_name, int start_line)
{
    std::ifstream semanticfile(file_name+"-semantic.txt", std::ios::in);
    if (semanticfile.is_open())
    {
        std::string line;
        size_t line_idx = 0;   
        while(std::getline(semanticfile, line))
        {
            if(line_idx == start_line)
            {
                // line 1, lable, conf, obs
                size_t pos=line.find(",");
                label = std::stoi(line.substr(0, pos));
                line.erase(0, pos+1);
                pos=line.find(",");
                confidence = std::stof(line.substr(0, pos));
                line.erase(0, pos+1);pos=line.find(",");
                observation = std::stoi(line.substr(0, pos));
                // std::cout << label << ", " << confidence << ", " << observation << std::endl;
                // std::cout << "label << confidence << observation" << std::endl;
                // line 2, pose
                std::getline(semanticfile, line);
                for(size_t r=0; r<4; r++){
                    for(size_t c=0; c<4; c++){
                        pos=line.find(",");
                        pose(r,c)=std::stod(line.substr(0, pos));
                        line.erase(0, pos+1);
                    }
                }
                // std::cout << "pose" << std::endl;
                // line 3, dims, scale
                std::getline(semanticfile, line);
                for(size_t idx=0; idx<3; ++idx){
                    pos=line.find(",");
                    dims.push_back( std::stof(line.substr(0, pos)) );
                    line.erase(0, pos+1);
                }
                pos=line.find(",");
                scale = std::stof(line.substr(0, pos));
                line.erase(0, pos+1);
                pos=line.find(",");
                sigma_scale = std::stod(line.substr(0, pos));
                // std::cout << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << scale << std::endl;
                // std::cout << "dims" << std::endl;
                // line 4, corners
                std::getline(semanticfile, line);
                cv::Mat corners(3, 8, CV_32FC1);
                for(size_t r=0; r<3; ++r){
                    for(size_t c=0; c<8; ++c){
                        pos=line.find(",");
                        corners.at<float>(r,c)=std::stof(line.substr(0, pos));
                        line.erase(0, pos+1);
                    }
                }
                cuboid_corner_pts = corners.clone();
                // std::cout << "cuboid_corner_pts" << std::endl;
                // line 5, axes
                std::getline(semanticfile, line);
                cv::Mat axis(3, 4, CV_32FC1);
                for(size_t r=0; r<3; ++r){
                    for(size_t c=0; c<4; ++c){
                        pos=line.find(",");
                        axis.at<float>(r,c)=std::stof(line.substr(0, pos));
                        line.erase(0, pos+1);
                    }
                }
                axes = axis.clone();
                // std::cout << "axes" << std::endl;
                // line 6, centroid
                std::getline(semanticfile, line);
                for(size_t r=0; r<3; r++){
                    pos=line.find(",");
                    centroid(r)=std::stod(line.substr(0, pos));
                    line.erase(0, pos+1);
                }
                // std::cout << "centroid" << std::endl;
                mean = centroid;
                // line 7, all class confidence
                std::getline(semanticfile, line);
                for(size_t i=0; i<7; ++i)
                {
                    pos=line.find(",");
                    all_class_confidence.push_back(std::stof(line.substr(0, pos)));
                    line.erase(0, pos+1);
                }
                // std::cout << "confidence" << std::endl;
                // line 8, MLE covariance
                std::getline(semanticfile, line);
                for(size_t r=0; r<3; ++r){
                    for(size_t c=0; c<3; ++c){
                        pos=line.find(",");
                        cov(r,c)=std::stof(line.substr(0, pos));
                        line.erase(0, pos+1);
                    }
                }
                // std::cout << "MLE cov" << std::endl;
                // line 9, propagated covariance
                std::getline(semanticfile, line);
                for(size_t r=0; r<3; ++r){
                    for(size_t c=0; c<3; ++c){
                        pos=line.find(",");
                        cov_propagated(r,c)=std::stof(line.substr(0, pos));
                        line.erase(0, pos+1);
                    }
                }
                // std::cout << "prop. cov" << std::endl;
                break;
            }
            line_idx++;
        }
    }
    semanticfile.close();
}

// std::vector<float> Cuboid3d::find_centroid(){
//     // cv::Mat cvMat;
//     // cv::reduce(data_pts, cvMat, 0, CV_REDUCE_AVG);
//     // // std::vector<float> ctr;
//     // std::vector<float> ctr{cvMat.at<float>(0), cvMat.at<float>(1), cvMat.at<float>(2)};
//     // return ctr;
// }
// // calculate centroid, and main axes
// void Cuboid3d::find_principal_axes(){
//     // cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
//     // // std::cout << "Eigenvectors are:\n" << pca_analysis.eigenvectors << std::endl;
//     // // std::cout << "Eigenvalues are:\n" << pca_analysis.eigenvalues << std::endl;
//     // // std::cout << "MEANS are:\n" << pca_analysis.mean << std::endl;
//     // // std::cout << pca_analysis.eigenvectors.type() << std::endl;
//     // for(size_t i=0; i<3; ++i){
//     //     // centroid
//     //     centroid[i] = pca_analysis.mean.at<float>(i);
//     //     // principal axes
//     //     Eigen::Vector3d point;
//     //     if(i < 2){
//     //         point << double(pca_analysis.eigenvectors.at<float>(i, 0)),
//     //                 double(pca_analysis.eigenvectors.at<float>(i, 1)),
//     //                 double(pca_analysis.eigenvectors.at<float>(i, 2));
//     //     } else {
//     //         // ENFORCE right-hand coordinate system
//     //         point = axes[0].cross(axes[1]);
//     //     }
//     //     axes[i] = point;
//     //     axes_length[i] = pca_analysis.eigenvalues.at<float>(i);
//     // }
//     // // std::cout << "Axes are: \n"
//     // //             << axes[0](0) << ", " << axes[0](1) << ", " << axes[0](2) << "; \n"
//     // //             << axes[1](0) << ", " << axes[1](1) << ", " << axes[1](2) << "; \n"
//     // //             << axes[2](0) << ", " << axes[2](1) << ", " << axes[2](2) << std::endl;
//     // // std::cout << "Axes' lengthes are: \n"
//     // //             << axes_length[0] << ", "
//     // //             << axes_length[1] << ", "
//     // //             << axes_length[2] << std::endl;
//     // // std::cout << "Centroid is: \n"
//     // //             << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;
// }
// bool Cuboid3d::find_bounding_cuboid(){
//     // // double check if data points are available
//     // if(data_pts.empty())
//     //     return false;
//     // // pca analysis for principal axes
//     // // std::clock_t start_0 = std::clock();
//     // find_principal_axes();
//     // // std::cout << "#### PCA analysis takes "
//     // //           << ( std::clock() - start_0 ) / (double) CLOCKS_PER_SEC
//     // //           << " seconds; ";
//     // // align principal axes with x, y, z
//     // bool b_pos_z = align_principal_axes();
//     // if(!b_pos_z)
//     //     std::cout << "LEFT-HAND coordinate system!!! STH WENT WRONG!!!" << std::endl;
//     // // rotate all points, find corner, and rotate corners back
//     // Eigen::Vector3d blb, trf;  // rotated coordinates
//     // for(size_t i=0; i<data_pts.rows; ++i)
//     // {
//     //     // rotate
//     //     Eigen::Vector3d point;
//     //     point << double(data_pts.at<float>(i, 0)), double(data_pts.at<float>(i, 1)), double(data_pts.at<float>(i, 2));
//     //     point = rotation*point;
//     //     // find the smallest corner and largest corner
//     //     if(i==0){
//     //         blb = point;
//     //         trf = point;
//     //         continue;
//     //     }
//     //     // x
//     //     if(blb(0) > point(0))
//     //         blb(0) = point(0);
//     //     if(trf(0) < point(0))
//     //         trf(0) = point(0);
//     //     // y
//     //     if(blb(1) > point(1))
//     //         blb(1) = point(1);
//     //     if(trf(1) < point(1))
//     //         trf(1) = point(1);
//     //     // z
//     //     if(blb(2) > point(2))
//     //         blb(2) = point(2);
//     //     if(trf(2) < point(2))
//     //         trf(2) = point(2);
//     // }
//     // // store blb & trf
//     // for(size_t j=0; j<3; ++j){
//     //     rotated_dim[j] = blb(j);
//     //     rotated_dim[j+3] = trf(j);
//     // }
//     // // find other corners
//     // std::vector<Eigen::Vector3d> vE_corners;
//     // Eigen::Vector3d brb, brf, blf, tlf, tlb, trb;
//     // vE_corners.push_back(blb);
//     // brb << blb(0), trf(1), blb(2);
//     // vE_corners.push_back(brb);
//     // brf << trf(0), trf(1), blb(2);
//     // vE_corners.push_back(brf);
//     // blf << trf(0), blb(1), blb(2);
//     // vE_corners.push_back(blf);
//     // tlf << trf(0), blb(1), trf(2);
//     // vE_corners.push_back(tlf);
//     // tlb << blb(0), blb(1), trf(2);
//     // vE_corners.push_back(tlb);
//     // trb << blb(0), trf(1), trf(2);
//     // vE_corners.push_back(trb);
//     // vE_corners.push_back(trf);
//     // // rotate back and store the corners
//     // // blb, brb, brf, blf; tlf, tlb, trb, trf.
//     // cuboid_corner_pts.clear();
//     // Eigen::Matrix3d rotation_inv = rotation.transpose();
//     // for(size_t i=0; i<8; ++i){
//     //     Eigen::Vector3d tmp_pt = rotation_inv*vE_corners[i];
//     //     cuboid_corner_pts.push_back(float(tmp_pt(0)));
//     //     cuboid_corner_pts.push_back(float(tmp_pt(1)));
//     //     cuboid_corner_pts.push_back(float(tmp_pt(2)));
//     // }
//     // // std::cout << "#### Find cuboids takes "
//     // //           << ( std::clock() - start_0 ) / (double) CLOCKS_PER_SEC
//     // //           << " seconds; \n";
//     // // std::cout << "#### The whole process takes "
//     // //           << ( std::clock() - start_0 ) / (double) CLOCKS_PER_SEC
//     // //           << " seconds; ";
//     // return true;
// }
// bool Cuboid3d::align_principal_axes() {
//     // // align principal axes with x, y, z
//     // // align firt axis with (1,0,0) with inhomogeneous coordinates
//     // cv::Vec3d targetx(1, 0, 0);
//     // cv::Vec3d currentx(axes[0](0), axes[0](1), axes[0](2));
// 	// Eigen::Matrix3d Rotx = calculate_rotation(currentx, targetx);
//     // // align the new second axis with (0,1,0)
//     // Eigen::Vector3d new_y_axis = Rotx * axes[1];
//     // cv::Vec3d targety(0, 1, 0);
//     // cv::Vec3d currenty(new_y_axis(0), new_y_axis(1), new_y_axis(2));
//     // Eigen::Matrix3d Roty = calculate_rotation(currenty, targety);
//     // // align the new second axis with (1,0,0)
//     // // Eigen::Vector3d new_z_axis = Roty * Rotx * axes[2];
//     // // cv::Vec3d targetz(0, 0, 1);
//     // // cv::Vec3d currentz(new_z_axis(0), new_z_axis(1), new_z_axis(2));
//     // // Eigen::Matrix3d Rotz = calculate_rotation(currentz, targetz);
//     // // Eigen::Matrix3d Rotz = Eigen::Matrix3d::Identity();
//     // rotation = Roty * Rotx;
//     // // // Check if Rotation matrix is correct.
//     // // std::cout << std::endl;
// 	// // std::cout << rotation*axes[0] << std::endl;
//     // // std::cout << rotation*axes[1] << std::endl;
//     // // std::cout << rotation*axes[2] << std::endl;
//     // // std::cout << std::endl;
//     // Eigen::Vector3d new_z = rotation*axes[2];
//     // return new_z(2)>0;
// }
// void Cuboid3d::merge_cuboids(std::shared_ptr<Cuboid3d> pNewCuboid){
//     // // store all map points
//     // // TODO: accummulate point every n keyframes instead of every 1 keyframe
//     // data_pts.push_back(pNewCuboid->data_pts);
//     // // update cuboid with all map points
//     // bool bCFound = find_bounding_cuboid();
//     // if(!bCFound)
//     //     std::cout << "UPDATED cuboid NOT FOUND!!!! SOMETHING is WRONG." << std::endl;
//     // // update confidence
//     // size_t prev_ob = observation;
//     // observation++;
//     // // TODO: figure a better way to update the probability
//     // confidence = (confidence * prev_ob + pNewCuboid->confidence)/(observation);
// }
// bool Cuboid3d::is_overlapped(std::shared_ptr<Cuboid3d> pNewCuboid){
//     // Eigen::Vector3d rot_ctr_pre = pNewCuboid->rotation * centroid;
//     // bool pre_in_new = rot_ctr_pre(0) >= pNewCuboid->rotated_dim[0] && rot_ctr_pre(0) <= pNewCuboid->rotated_dim[3]
//     //                && rot_ctr_pre(1) >= pNewCuboid->rotated_dim[1] && rot_ctr_pre(1) <= pNewCuboid->rotated_dim[4]
//     //                && rot_ctr_pre(2) >= pNewCuboid->rotated_dim[2] && rot_ctr_pre(2) <= pNewCuboid->rotated_dim[5];
//     // Eigen::Vector3d rot_ctr_new = rotation * pNewCuboid->centroid;
//     // bool new_in_pre = rot_ctr_new(0) >= rotated_dim[0] && rot_ctr_new(0) <= rotated_dim[3]
//     //                && rot_ctr_new(1) >= rotated_dim[1] && rot_ctr_new(1) <= rotated_dim[4]
//     //                && rot_ctr_new(2) >= rotated_dim[2] && rot_ctr_new(2) <= rotated_dim[5];
//     // // std::cout << "Overlap: " << label << "-" << pNewCuboid->label << ": " << pre_in_new << ", " << new_in_pre << std::endl;
//     // return (pre_in_new || new_in_pre);
// }
// void Cuboid3d::update_label(std::shared_ptr<Cuboid3d> pNewCuboid){
//     // // compare the confidence
//     // float conf_pre = (confidence * observation + 0.5)/(observation + 1);
//     // float conf_new = (0.5 * observation + pNewCuboid->confidence)/(observation + 1);
//     // std::cout << label << "-" << conf_pre << " v.s. "
//     //             << pNewCuboid->label << "-" << conf_new << ": ";
//     // if(conf_pre >= conf_new)
//     // // update confidence and discard new detection
//     // {
//     //     std::cout << "KEEP the previous label." << std::endl;
//     //     confidence = conf_pre;
//     // }
//     // else
//     // // update paras in pre with paras in new
//     // {
//     //     std::cout << "UPDATE with the new label." << std::endl;
//     //     // update label
//     //     label = pNewCuboid->label;
//     //     confidence = conf_new;
//     //     observation = 1;
//     //     // update cuboid
//     //     data_pts.push_back(pNewCuboid->data_pts);
//     //     // update cuboid with all map points
//     //     bool bCFound = find_bounding_cuboid();
//     //     if(!bCFound)
//     //         std::cout << "UPDATED cuboid NOT FOUND!!!! SOMETHING is WRONG." << std::endl;
//     // }
// }
// void Cuboid3d::fit_cuboid(std::shared_ptr<Cuboid3d> pTargetCuboid, int mask_width){
//     // // std::cout << "# of valid map point is: " << data_pts.rows << std::endl;
//     // // filter out outliers
//     // // if a point has less than x points in its neighborhood, it is considered as an outlier
//     // size_t num_map_pts = data_pts.rows;
//     // cv::Mat map_pts = data_pts.clone();
//     // data_pts.release();
//     // size_t thre_neighbour = 30;
//     // double thre_dist = 0.01;        // 0.01 meters
//     // for(size_t i=0; i<num_map_pts; ++i)
//     // {
//     //     float test_pt_x = map_pts.at<float>(i,0),
//     //           test_pt_y = map_pts.at<float>(i,1),
//     //           test_pt_z = map_pts.at<float>(i,2);
//     //     size_t neighbour_count = 0;
//     //     // double dist_min = 100, dist_max= 0;
//     //     // for(size_t j=0; j<num_map_pts; ++j)
//     //     for(size_t j= i-mask_width*5; j<i+mask_width*5; ++j)
//     //     {
//     //         if(j == i || j<0 || j>=num_map_pts)
//     //             continue;
//     //         double dist = std::sqrt(
//     //             std::pow(double(test_pt_x - map_pts.at<float>(j,0)), 2) +
//     //             std::pow(double(test_pt_y - map_pts.at<float>(j,1)), 2) +
//     //             std::pow(double(test_pt_z - map_pts.at<float>(j,2)), 2)
//     //         );
//     //         // dist_min = dist < dist_min ? dist : dist_min;
//     //         // dist_max = dist > dist_max ? dist : dist_max;
//     //         if(dist < thre_dist)
//     //             neighbour_count ++;
//     //     }
//     //     // std::cout << dist_min << ", " << dist_max << std::endl;
//     //     // break;
//     //     if(neighbour_count > thre_neighbour){
//     //         // accept the map point
//     //         cv::Mat cvMat(1, 3, CV_32F);
//     //         cvMat.at<float>(0,0) = test_pt_x;
//     //         cvMat.at<float>(0,1) = test_pt_y;
//     //         cvMat.at<float>(0,2) = test_pt_z;
//     //         data_pts.push_back(cvMat);
//     //     }
//     // }
//     // std::cout << "# of pts before filtering: " << num_map_pts
//     //           << "; after filtering: " << data_pts.rows << std::endl;
//     // // find object as did before ////
//     // find_bounding_cuboid();
//     // // find all possible arrangements of corner points
//     // // idx: correct pt order; val: corresponding pt in the cuboid
//     // int idx[10][8] = {
//     //     {0, 1, 2, 3, 4, 5, 6, 7},
//     //     {1, 6, 7, 2, 3, 0, 5, 4},
//     //     {6, 5, 4, 7, 2, 1, 0, 3},
//     //     {5, 0, 3, 4, 7, 6, 1, 2},
//     //     {3, 2, 7, 4, 5, 0, 1, 6},
//     //     {4, 7, 6, 5, 0, 3, 2, 1},
//     //     {5, 6, 1, 0, 3, 4, 7, 2},
//     //     {1, 2, 3, 0, 5, 6, 7, 4},
//     //     {2, 3, 0, 1, 6, 7, 4, 5},
//     //     {3, 0, 1, 2, 7, 4, 5, 6}
//     // };
//     // // the other way around; turns out be the same just in different order
//     //     // int idx[10][8] = {
//     //     //     {0, 1, 2, 3, 4, 5, 6, 7},
//     //     //     {5, 0, 3, 4, 7, 6, 1, 2},
//     //     //     {6, 5, 4, 7, 2, 1, 0, 3},
//     //     //     {1, 6, 7, 2, 3, 0, 5, 4},
//     //     //     {5, 6, 1, 0, 3, 4, 7, 2},
//     //     //     {4, 7, 6, 5, 0. 3, 2, 1},
//     //     //     {3, 2, 7, 4, 5, 0, 1, 6},
//     //     //     {3, 0, 1, 2, 7, 4, 5, 6},
//     //     //     {2, 3, 0, 1, 6, 7, 4, 5},
//     //     //     {1, 2, 3, 0, 5, 6, 7, 4}
//     // // };
//     // for(size_t i=0; i<10; ++i)
//     // {
//     //     std::vector<float> candidate;
//     //     for(size_t j=0; j<8; ++j)
//     //     {
//     //         candidate.push_back(cuboid_corner_pts[idx[i][j]*3]);
//     //         candidate.push_back(cuboid_corner_pts[idx[i][j]*3+1]);
//     //         candidate.push_back(cuboid_corner_pts[idx[i][j]*3+2]);
//     //     }
//     //     corner_candidates.push_back(candidate);
//     // }
//     // // std::cout << "#### When storing the corners ##########" << std::endl;
//     // // for(size_t i=0; i<10; ++i)
//     // // {
//     // //     std::cout << "Configuration #" << i << ":" << std::endl;
//     // //     for(size_t j=0; j<8; ++j)
//     // //     {
//     // //         std::cout << j << "-("
//     // //                   << corner_candidates[i][j*3] << ", "
//     // //                   << corner_candidates[i][j*3+1] << ", "
//     // //                   << corner_candidates[i][j*3+2] << "); "
//     // //                   << std::endl;
//     // //     }
//     // // }
//     // // std::cout << "#########################################" << std::endl;
//     // // /////// OR ///////
//     // // // fit with detected size ////
//     // // // find centroid and main axes
//     // // find_principal_axes();
//     // // // find rotation
//     // // bool b_pos_z = align_principal_axes();
//     // // if(!b_pos_z)
//     // //     std::cout << "LEFT-HAND coordinate system!!! STH WENT WRONG!!!" << std::endl;
//     // // // fit current point cloud to the cuboid
//     // // translate_corner_pts(pTargetCuboid);
//     // // /////// OR ///////
//     // // // TODO:
//     // // // find better way to fit cuboid
//     // // // i.e. maximize the # of pts inside the cuboid
// }
// void Cuboid3d::translate_corner_pts(std::shared_ptr<Cuboid3d> pTargetCuboid){
//     // // rotate back and store the corners
//     // Eigen::Matrix3d overall_rot = rotation.transpose() * pTargetCuboid->rotation;
//     // cuboid_corner_pts.clear();
//     // std::vector<float> candidate_1;
//     // // std::vector<double> tar_cor_cen_dist, can1_cor_cen_dist;
//     // // double loss_1 = 0;
//     // for(size_t i=0; i<8; ++i){
//     //     Eigen::Vector3d tarPt(pTargetCuboid->cuboid_corner_pts[ i*3 ],
//     //                           pTargetCuboid->cuboid_corner_pts[ i*3+1 ],
//     //                           pTargetCuboid->cuboid_corner_pts[ i*3+2 ]);
//     //     Eigen::Vector3d curPt = overall_rot * (tarPt - pTargetCuboid->centroid) + centroid;
//     //     // tar_cor_cen_dist.push_back( (tarPt - pTargetCuboid->centroid).norm() );
//     //     // can1_cor_cen_dist.push_back( (curPt - centroid).norm() );
//     //     // loss_1 += (curPt - centroid).norm() ;
//     //     // Eigen::Vector3d curPt = overall_rot * (tarPt - tar_cub_centroid) + cur_cub_centroid;
//     //     candidate_1.push_back(float(curPt(0)));
//     //     candidate_1.push_back(float(curPt(1)));
//     //     candidate_1.push_back(float(curPt(2)));
//     // }
//     // // std::vector<float> candidate_2(24);
//     // // double loss_2 = 0;
//     // // int idx[8] = {2, 3, 0, 1, 6, 7, 4, 5};
//     // // for(size_t i=0; i<8; ++i){
//     // //     Eigen::Vector3d tarPt(pTargetCuboid->cuboid_corner_pts[ i*3 ],
//     // //                           pTargetCuboid->cuboid_corner_pts[ i*3+1 ],
//     // //                           pTargetCuboid->cuboid_corner_pts[ i*3+2 ]);
//     // //     Eigen::Vector3d curPt = overall_rot * (tarPt - pTargetCuboid->centroid) + centroid;
//     // //     // std::cout << "  " << tar_cor_cen_dist[i] << ", "
//     // //     //           << can1_cor_cen_dist[i] << ", "
//     // //     //           << (curPt - centroid).norm() << std::endl;
//     // //     // loss_2 += (curPt - centroid).norm() ;
//     // //     // candidate_2.push_back(float(curPt(0)));
//     // //     // candidate_2.push_back(float(curPt(1)));
//     // //     // candidate_2.push_back(float(curPt(2)));
//     // //     candidate_2[idx[i]*3] = float(curPt(0));
//     // //     candidate_2[idx[i]*3+1] = float(curPt(1));
//     // //     candidate_2[idx[i]*3+2] = float(curPt(2));
//     // // }
//     // // std::cout << loss_1 << " vs. " << loss_2 << std::endl;
//     // // if(loss_1 > loss_2){
//     //     cuboid_corner_pts = candidate_1;
//     // // } else {
//     // //     cuboid_corner_pts = candidate_2;
//     // // }
// }

} // namespace fusion