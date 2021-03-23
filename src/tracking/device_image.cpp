#include "tracking/device_image.h"
#include "tracking/cuda_imgproc.h"
#include <ctime>

namespace fusion
{

DeviceImage::DeviceImage(const std::vector<Eigen::Matrix3f> vIntrinsicsInv)
{
    vKInv = vIntrinsicsInv;
    // NUM_PYRS = vKInv.size();
}

DeviceImage::DeviceImage(const DeviceImage &other)
{
    depth_pyr = other.depth_pyr;
    intensity_pyr = other.intensity_pyr;
    intensity_dx_pyr = other.intensity_dx_pyr;
    intensity_dy_pyr = other.intensity_dy_pyr;
    vmap_pyr = other.vmap_pyr;
    nmap_pyr = other.nmap_pyr;
    semi_dense_image = other.semi_dense_image;
    vKInv = other.vKInv;
    image = other.image;
    image_float = other.image_float;
    depth_float = other.depth_float;
    intensity_float = other.intensity_float;
    rendered_image = other.rendered_image;
    rendered_image_textured = other.rendered_image_textured;

    // cent_float = other.cent_float;
    cent_float_cpu = other.cent_float_cpu.clone();

    detected_masks = other.detected_masks;
    v_detected_bboxes = other.v_detected_bboxes;
}

DeviceImage &DeviceImage::operator=(DeviceImage other)
{
    if (this != &other)
    {
        swap(*this, other);
    }
}

void swap(DeviceImage &first, DeviceImage &second)
{
    // change later // 
    first.image.swap(second.image);
    first.image_float.swap(second.image_float);
    first.depth_float.swap(second.depth_float);
    first.intensity_float.swap(second.intensity_float);
    // first.cent_float.swap(second.cent_float);
    cv::swap(first.cent_float_cpu, second.cent_float_cpu);
    
    std::swap(first.depth_pyr, second.depth_pyr);
    std::swap(first.intensity_pyr, second.intensity_pyr);
    std::swap(first.intensity_dx_pyr, second.intensity_dx_pyr);
    std::swap(first.intensity_dy_pyr, second.intensity_dy_pyr);
    std::swap(first.vmap_pyr, second.vmap_pyr);
    std::swap(first.nmap_pyr, second.nmap_pyr);
    std::swap(first.semi_dense_image, second.semi_dense_image);
    first.rendered_image.swap(second.rendered_image);
    first.rendered_image_textured.swap(second.rendered_image_textured);
}

void copyDeviceImage(RgbdImagePtr src, RgbdImagePtr dst)
{
    // dst->reference_frame = std::make_shared<RgbdFrame>();
    // src->reference_frame->copyTo(dst->reference_frame);
    dst->reference_frame = src->reference_frame;    // src->ref has a new ptr for new input

    src->image.copyTo(dst->image);
    src->image_float.copyTo(dst->image_float);
    src->depth_float.copyTo(dst->depth_float);
    src->intensity_float.copyTo(dst->intensity_float);
    // src->cent_float.copyTo(dst->cent_float);
    dst->cent_float_cpu = src->cent_float_cpu.clone();

    size_t level = src->depth_pyr.size();
    if (dst->depth_pyr.size() != level){
        dst->depth_pyr.resize(level);
        dst->intensity_pyr.resize(level);
        dst->intensity_dx_pyr.resize(level);
        dst->intensity_dy_pyr.resize(level);
        dst->vmap_pyr.resize(level);
        dst->nmap_pyr.resize(level);

        dst->semi_dense_image.resize(level);
    }
    for(size_t i=0; i<level; ++i){
        src->depth_pyr[i].copyTo(dst->depth_pyr[i]);
        src->intensity_pyr[i].copyTo(dst->intensity_pyr[i]);
        src->intensity_dx_pyr[i].copyTo(dst->intensity_dx_pyr[i]);
        src->intensity_dy_pyr[i].copyTo(dst->intensity_dy_pyr[i]);
        src->vmap_pyr[i].copyTo(dst->vmap_pyr[i]);
        src->nmap_pyr[i].copyTo(dst->nmap_pyr[i]);
        
        // src->semi_dense_image[i].copyTo(dst->semi_dense_image[i]);  // PROBLEM here!! CHECK where this is used !!!!!!!!!!!!!!!!
        
        // // calculate cam_params while initializing
        // cv::cuda::GpuMat tmp_k;
        // src->cam_params[i].copyTo(tmp_k);
        // dst->cam_params.push_back(tmp_k);
    }

    // src->rendered_image.copyTo(dst->rendered_image);     // ALSO CHECK here !!!!!!!!!
    // src->rendered_image_textured.copyTo(dst->rendered_image_textured);
    // src->rendered_image_raw.copyTo(dst->rendered_image_raw);

    src->detected_masks.copyTo(dst->detected_masks);
    if(dst->v_detected_bboxes.size() != src->v_detected_bboxes.size())
        dst->v_detected_bboxes.resize(src->v_detected_bboxes.size());
    for(size_t i=0; i<src->v_detected_bboxes.size(); ++i)
        src->v_detected_bboxes[i].copyTo(dst->v_detected_bboxes[i]);
}

void DeviceImage::resize_pyramid(const int &max_level)
{
    depth_pyr.resize(max_level);
    intensity_pyr.resize(max_level);
    intensity_dx_pyr.resize(max_level);
    intensity_dy_pyr.resize(max_level);
    vmap_pyr.resize(max_level);
    nmap_pyr.resize(max_level);
}

void DeviceImage::create_depth_pyramid(const int max_level, const bool use_filter)
{
    if (depth_float.empty() || depth_float.type() != CV_32FC1)
    {
        std::cout << "depth not supplied." << std::endl;
    }

    if (depth_pyr.size() != max_level)
        depth_pyr.resize(max_level);

    if (use_filter)
        filterDepthBilateral(depth_float, depth_pyr[0]);
    else
        depth_float.copyTo(depth_pyr[0]);

    for (int level = 1; level < max_level; ++level)
    {
        // cv::cuda::resize(depth_pyr[level - 1], depth_pyr[level], cv::Size(0, 0), 0.5, 0.5);
        pyrDownDepth(depth_pyr[level - 1], depth_pyr[level]);
    }
}

void DeviceImage::create_intensity_pyramid(const int max_level)
{
    if (intensity_pyr.size() != max_level)
        intensity_pyr.resize(max_level);

    intensity_float.copyTo(intensity_pyr[0]);

    for (int level = 1; level < max_level; ++level)
    {
        // cv::cuda::pyrDown(intensity_pyr[level - 1], intensity_pyr[level]);
        pyrDownImage(intensity_pyr[level - 1], intensity_pyr[level]);
    }
}

void DeviceImage::create_vmap_pyramid(const int max_level)
{
}

void DeviceImage::create_nmap_pyramid(const int max_level)
{
}

cv::cuda::GpuMat &DeviceImage::get_vmap(const int &level)
{
    return vmap_pyr[level];
}

cv::cuda::GpuMat &DeviceImage::get_nmap(const int &level)
{
    return nmap_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_rendered_image()
{
    renderScene(vmap_pyr[0], nmap_pyr[0], rendered_image);
    return rendered_image;
}

cv::cuda::GpuMat DeviceImage::get_rendered_scene_textured()
{
    renderSceneTextured(vmap_pyr[0], nmap_pyr[0], image, rendered_image);
    return rendered_image;
}

cv::cuda::GpuMat DeviceImage::get_intensity(const int &level) const
{
    return intensity_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_intensity_dx(const int &level) const
{
    return intensity_dx_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_intensity_dy(const int &level) const
{
    return intensity_dy_pyr[level];
}

cv::Mat DeviceImage::get_centroids() const
{
    return cent_float_cpu;
}

void DeviceImage::upload(const std::shared_ptr<RgbdFrame> frame)
{
    if (frame == reference_frame){
        std::cout << "this frame already uploaded" << std::endl;
        return;
    }
    // std::cout << "Uploading frame" << std::endl;

    const int max_level = vKInv.size();

    if (max_level != this->depth_pyr.size())
        resize_pyramid(max_level);

    cv::Mat image = frame->image;
    cv::Mat depth = frame->depth;

    this->image.upload(image);
    depth_float.upload(depth);
    this->image.convertTo(image_float, CV_32FC3);
    cv::cuda::cvtColor(image_float, intensity_float, cv::COLOR_RGB2GRAY);

    create_depth_pyramid(max_level);
    create_intensity_pyramid(max_level);

    if (intensity_dx_pyr.size() != max_level)
        intensity_dx_pyr.resize(max_level);

    if (intensity_dy_pyr.size() != max_level)
        intensity_dy_pyr.resize(max_level);

    for (int i = 0; i < max_level; ++i)
    {
        computeDerivative(intensity_pyr[i], intensity_dx_pyr[i], intensity_dy_pyr[i]);
        backProjectDepth(depth_pyr[i], vmap_pyr[i], vKInv[i]);
        computeNMap(vmap_pyr[i], nmap_pyr[i]);
    }

    reference_frame = frame;

    cent_float_cpu = frame->cent_matrix.clone();
    // this->cent_float.upload(cent_float_cpu);


    // // !!!!! COMMENT this line out later
    // renderScene(vmap_pyr[0], nmap_pyr[0], rendered_image_raw);
}

cv::cuda::GpuMat DeviceImage::get_rendered_image_raw()
{
    if(rendered_image_raw.empty())
        std::cout << "raw rendered scene is empty" << std::endl;
    return rendered_image_raw;
}

// DeviceImage::DeviceImage(const int &max_level)
// {
//     resize_pyramid(max_level);
// }

void DeviceImage::resize_device_map()
{
    // re-calculate v & n pyramid given rendered 
    for (int i = 1; i < vmap_pyr.size(); ++i)
    {
        pyrDownVMap(vmap_pyr[i - 1], vmap_pyr[i]);
    }

    for (int i = 0; i < vmap_pyr.size(); ++i)
    {
        computeNMap(vmap_pyr[i], nmap_pyr[i]);
    }
}

void DeviceImage::downloadVNM(RgbdFramePtr frame, bool bTrackLost)
{
    // download rendered v & n at level 0 to ref frame
    if(!vmap_pyr[0].empty()){
        vmap_pyr[0].download(frame->vmap);
        // update depth map
        std::vector<cv::Mat> channels;
        cv::split(frame->vmap, channels);
        if(!bTrackLost)
            frame->depth = channels[2].clone();
    } else {
        std::cout << "-- EMPTY VMAP --" << std::endl;
    }
    if(!nmap_pyr[0].empty()){
        nmap_pyr[0].download(frame->nmap);
    } else {
        std::cout << "-- EMPTY NMAP --" << std::endl;
    }
    // if(!detected_masks.empty()){
    //     detected_masks.download(frame->mask);
    // }
    // else{
    //     std::cout << "-- EMPTY MASK --"  << std::endl;
    // }
    // vmap_pyr[0].download(reference_frame->vmap);
    // nmap_pyr[0].download(reference_frame->nmap);
    if( frame->vmap.empty() || frame->nmap.empty() || (frame->mask.empty() && !detected_masks.empty()) )
        std::cout << "!!!!!!! download failed !!!!!!!" << std::endl;
}

void DeviceImage::upload_semantics(const std::shared_ptr<RgbdFrame> frame)
{
    // std::cout << "Calling \"upload_semantics\" but nothing defined here." << std::endl;
    detected_masks.upload(frame->mask);
    // detected_bboxes.upload(frame->bbox);
    // for(size_t i=0; i<frame->numDetection; ++i){
    //     cv::cuda::GpuMat one_bbox(1, 4, CV_32F);
    //     one_bbox.upload(frame->v_bbox[i]);
    //     v_detected_bboxes.push_back(one_bbox);
    // }

    // std::cout << "size of GpuMat: " << v_detected_bboxes.size() << std::endl; 
}

void DeviceImage::GeometricRefinement(float lamb, float tao, int win_size, cv::Mat &edge_device)
{
    std::clock_t start = std::clock();
    cv::cuda::GpuMat edge(image.size(), CV_8UC1);
    int step = win_size/2;

    // compute edge map
    NVmapToEdge(nmap_pyr[0], vmap_pyr[0], edge, lamb, tao, win_size, step);
    cv::Mat tmp_edge(edge_device.size(), CV_8UC1);
    edge.download(tmp_edge);

    // add morphology operation to get better edge map
    cv::Mat morph_ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
    cv::morphologyEx(tmp_edge, edge_device, cv::MORPH_OPEN, morph_ker, cv::Point(1,1), 1);

    std::cout << "#### Geometric Segmentation takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << " seconds" << std::endl;

    // cv::imwrite("/home/yohann/edge_original.png", tmp_edge);
    // cv::imwrite("/home/yohann/edge_openned.png", edge_device);
}

cv::cuda::GpuMat &DeviceImage::get_object_mask()
{
    if(!detected_masks.empty())
        return detected_masks;
}

cv::cuda::GpuMat DeviceImage::get_object_bbox(const int &i) const
{
    if(i < v_detected_bboxes.size())
        return v_detected_bboxes[i];
}

int DeviceImage::get_object_label(const int &i) const
{
    if(i < reference_frame->vLabels.size())
        return reference_frame->vLabels[i];
}

cv::cuda::GpuMat DeviceImage::get_depth(const int &level) const
{
    if (level < depth_pyr.size())
        return depth_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_raw_depth() const
{
    return depth_float;
}

cv::cuda::GpuMat DeviceImage::get_image() const
{
    return image;
}

RgbdFramePtr DeviceImage::get_reference_frame() const
{
    return reference_frame;
}

} // namespace fusion