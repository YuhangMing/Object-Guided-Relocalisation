#include "tracking/pose_estimator.h"
#include "tracking/reduce_sum.h"
#include "utils/safe_call.h"
#include "utils/math/matrix_type.h"
#include "utils/math/vector_type.h"
#include <thrust/device_vector.h>

namespace fusion
{

struct RgbReduction
{
    FUSION_DEVICE inline bool find_corresp(int &x, int &y)
    {
        Vector4f pt = last_vmap.ptr(y)[x];
        if (pt.w < 0 || isnan(pt.x))
            return false;

        i_l = last_image.ptr(y)[x];
        if (!isfinite(i_l))
            return false;

        p_transformed = pose(ToVector3(pt));
        u0 = p_transformed.x / p_transformed.z * fx + cx;
        v0 = p_transformed.y / p_transformed.z * fy + cy;
        if (u0 >= 2 && u0 < cols - 2 && v0 >= 2 && v0 < rows - 2)
        {
            i_c = interp2(curr_image, u0, v0);
            dx = interp2(dIdx, u0, v0);
            dy = interp2(dIdy, u0, v0);

            return (dx > 2 || dy > 2) && isfinite(i_c) && isfinite(dx) && isfinite(dy);
        }

        return false;
    }

    FUSION_DEVICE inline float interp2(cv::cuda::PtrStep<float> image, float &x, float &y)
    {
        int u = std::floor(x), v = std::floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    FUSION_DEVICE inline void compute_jacobian(int &k, float *sum)
    {
        int y = k / cols;
        int x = k - y * cols;

        bool corresp_found = find_corresp(x, y);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (corresp_found)
        {
            Vector3f left;
            float z_inv = 1.0 / p_transformed.z;
            left.x = dx * fx * z_inv;
            left.y = dy * fy * z_inv;
            left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;
            row[6] = i_l - i_c;

            *(Vector3f *)&row[0] = left;
            // *(Vector3f *)&row[3] = p_transformed.cross(left);
            row[3] = row[2] * p_transformed.y - dy * fy;
            row[4] = -row[2] * p_transformed.x + dx * fx;
            row[5] = -row[0] * p_transformed.y + row[1] * p_transformed.x;
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];

        sum[count] = (float)corresp_found;
    }

    FUSION_DEVICE inline void operator()()
    {
        float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        float val[29];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            compute_jacobian(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        BlockReduce<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }

    int cols, rows, N;
    float u0, v0;
    Matrix3x4f pose;
    float fx, fy, cx, cy, invfx, invfy;
    cv::cuda::PtrStep<Vector4f> point_cloud, last_vmap;
    cv::cuda::PtrStep<float> last_image, curr_image;
    cv::cuda::PtrStep<float> dIdx, dIdy;
    cv::cuda::PtrStep<float> out;
    Vector3f p_transformed, p_last;

private:
    float i_c, i_l, dx, dy;
};

__global__ void rgb_reduce_kernel(RgbReduction rr)
{
    rr();
}

void rgb_reduce(
    const cv::cuda::GpuMat &curr_intensity,
    const cv::cuda::GpuMat &last_intensity,
    const cv::cuda::GpuMat &last_vmap,
    const cv::cuda::GpuMat &curr_vmap,
    const cv::cuda::GpuMat &intensity_dx,
    const cv::cuda::GpuMat &intensity_dy,
    cv::cuda::GpuMat &sum,
    cv::cuda::GpuMat &out,
    const Sophus::SE3d &pose,
    const Eigen::Matrix3f K,
    float *jtj, float *jtr,
    float *residual)
{
    int cols = curr_intensity.cols;
    int rows = curr_intensity.rows;

    RgbReduction rr;
    rr.cols = cols;
    rr.rows = rows;
    rr.N = cols * rows;
    rr.last_image = last_intensity;
    rr.curr_image = curr_intensity;
    rr.point_cloud = curr_vmap;
    rr.last_vmap = last_vmap;
    rr.dIdx = intensity_dx;
    rr.dIdy = intensity_dy;
    rr.pose = Matrix3x4f(pose.cast<float>().matrix3x4());
    rr.fx = K(0,0);
    rr.fy = K(1,1);
    rr.cx = K(0,2);
    rr.cy = K(1,2);
    rr.invfx = 1.0 / rr.fx;
    rr.invfy = 1.0 / rr.fy;
    rr.out = sum;

    rgb_reduce_kernel<<<96, 224>>>(rr);
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = host_data.ptr<float>()[28];
}

struct RgbReduction2
{
    __device__ bool find_corresp(int &x, int &y)
    {
        Vector4f pt = last_vmap.ptr(y)[x];
        if (pt.w < 0 || isnan(pt.x))
            return false;

        i_l = last_image.ptr(y)[x];
        if (!isfinite(i_l))
            return false;

        p_transformed = pose(ToVector3(pt));
        u0 = p_transformed.x / p_transformed.z * fx + cx;
        v0 = p_transformed.y / p_transformed.z * fy + cy;
        if (u0 >= 2 && u0 < cols - 2 && v0 >= 2 && v0 < rows - 2)
        {
            i_c = interp2(curr_image, u0, v0);
            dx = interp2(dIdx, u0, v0);
            dy = interp2(dIdy, u0, v0);

            return (dx > 2 || dy > 2) && isfinite(i_c) && isfinite(dx) && isfinite(dy);
        }

        return false;
    }

    __device__ float interp2(cv::cuda::PtrStep<float> image, float &x, float &y)
    {
        int u = std::floor(x), v = std::floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    __device__ void compute_jacobian(int &k, float *sum)
    {
        int y = k / cols;
        int x = k - y * cols;

        bool corresp_found = find_corresp(x, y);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (corresp_found)
        {
            Vector3f left;
            float z_inv = 1.0 / p_transformed.z;
            left.x = dx * fx * z_inv;
            left.y = dy * fy * z_inv;
            left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;

            float residual = i_c - i_l;

            if (stddev > 10e-5)
                residual /= stddev;

            float huber_th = 1.345 * stddev;

            float weight = 1;

            if (fabs(residual) > huber_th && stddev > 10e-6)
            {
                weight = sqrtf(huber_th / fabs(residual));
            }

            row[6] = weight * (-residual);
            *(Vector3f *)&row[0] = weight * left;
            // *(Vector3f *)&row[3] = weight * p_transformed.cross(left);
            // row[0] = dx * fx * z_inv;
            // row[1] = dy * fy * z_inv;
            // row[2] = -(row[0] * p_transformed.x + row[1] * p_transformed.y) * z_inv;
            row[3] = row[2] * p_transformed.y - dy * fy;
            row[4] = -row[2] * p_transformed.x + dx * fx;
            row[5] = -row[0] * p_transformed.y + row[1] * p_transformed.x;
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];

        sum[count] = (float)corresp_found;
    }

    __device__ __forceinline__ void operator()()
    {
        float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        float val[29];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            compute_jacobian(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        BlockReduce<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }

    int cols, rows, N;
    float u0, v0;
    Matrix3x4f pose;
    float fx, fy, cx, cy, invfx, invfy;
    cv::cuda::PtrStep<Vector4f> point_cloud, last_vmap;
    cv::cuda::PtrStep<float> last_image, curr_image;
    cv::cuda::PtrStep<float> dIdx, dIdy;
    cv::cuda::PtrStep<float> out;
    Vector3f p_transformed, p_last;
    float stddev;

private:
    float i_c, i_l, dx, dy;
};

__global__ void rgb_reduce_kernel2(RgbReduction2 rr)
{
    rr();
}

void rgb_step(const cv::cuda::GpuMat &curr_intensity,
              const cv::cuda::GpuMat &last_intensity,
              const cv::cuda::GpuMat &last_vmap,
              const cv::cuda::GpuMat &curr_vmap,
              const cv::cuda::GpuMat &intensity_dx,
              const cv::cuda::GpuMat &intensity_dy,
              cv::cuda::GpuMat &sum,
              cv::cuda::GpuMat &out,
              const float stddev_estimate,
              const Sophus::SE3d &pose,
              const Eigen::Matrix3f K,
              float *jtj, float *jtr,
              float *residual)
{
    int cols = curr_intensity.cols;
    int rows = curr_intensity.rows;

    RgbReduction2 rr;
    rr.cols = cols;
    rr.rows = rows;
    rr.N = cols * rows;
    rr.last_image = last_intensity;
    rr.curr_image = curr_intensity;
    rr.point_cloud = curr_vmap;
    rr.last_vmap = last_vmap;
    rr.dIdx = intensity_dx;
    rr.dIdy = intensity_dy;
    rr.pose = pose.cast<float>().matrix3x4();
    rr.stddev = stddev_estimate;
    rr.fx = K(0,0);
    rr.fy = K(1,1);
    rr.cx = K(0,2);
    rr.cy = K(1,2);
    rr.invfx = 1.0 / rr.fx;
    rr.invfy = 1.0 / rr.fy;
    rr.out = sum;

    rgb_reduce_kernel2<<<96, 224>>>(rr);
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = host_data.ptr<float>()[28];
}

struct ICPReduction
{
    FUSION_DEVICE inline bool searchPoint(int &x, int &y, Vector3f &vcurr_g, Vector3f &vlast_g, Vector3f &nlast_g) const
    {
        Vector3f vlast_c = ToVector3(last_vmap_.ptr(y)[x]);
        if (isnan(vlast_c.x))
            return false;

        vlast_g = pose(vlast_c);

        float invz = 1.0 / vlast_g.z;
        int u = __float2int_rd(vlast_g.x * invz * fx + cx + 0.5);
        int v = __float2int_rd(vlast_g.y * invz * fy + cy + 0.5);
        if (u < 0 || v < 0 || u >= cols || v >= rows)
            return false;

        vcurr_g = ToVector3(curr_vmap_.ptr(v)[u]);

        Vector3f nlast_c = ToVector3(last_nmap_.ptr(y)[x]);
        nlast_g = pose.rotate(nlast_c);

        Vector3f ncurr_g = ToVector3(curr_nmap_.ptr(v)[u]);

        float dist = (vlast_g - vcurr_g).norm();
        float sine = ncurr_g.cross(nlast_g).norm();

        return (sine < angleThresh && dist <= distThresh && !isnan(ncurr_g.x) && !isnan(nlast_g.x));
    }

    FUSION_DEVICE inline void getRow(int &i, float *sum) const
    {
        int y = i / cols;
        int x = i - y * cols;

        bool found = false;
        Vector3f vcurr, vlast, nlast;
        found = searchPoint(x, y, vcurr, vlast, nlast);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (found)
        {
            *(Vector3f *)&row[0] = nlast;
            *(Vector3f *)&row[3] = vlast.cross(nlast);
            row[6] = nlast * (vcurr - vlast);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
        {
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];
        }

        sum[count] = (float)found;
    }

    FUSION_DEVICE inline void operator()() const
    {
        float sum[29] = {0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0};

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float val[29];
        for (; i < N; i += blockDim.x * gridDim.x)
        {
            getRow(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        // pre-fix sum scan
        // calculate summation of all 29 elements held in all the threads.
        BlockReduce<float, 29>(sum);

        // again, summed results are stored in the very first thread in first block
        // store summed results into "out", which corresponded to the "sum" passed in
        // regarding which block it's in, store in the corresponding row
        if (threadIdx.x == 0)
        {
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
        }
    }

    Matrix3x4f pose;
    cv::cuda::PtrStep<Vector4f> curr_vmap_, last_vmap_;
    cv::cuda::PtrStep<Vector4f> curr_nmap_, last_nmap_;
    int cols, rows, N;
    float fx, fy, cx, cy;
    float angleThresh, distThresh;
    mutable cv::cuda::PtrStepSz<float> out;
};

__global__ void icp_reduce_kernel(const ICPReduction icp)
{
    icp();
}

void icp_reduce(
    const cv::cuda::GpuMat &curr_vmap,
    const cv::cuda::GpuMat &curr_nmap,
    const cv::cuda::GpuMat &last_vmap,
    const cv::cuda::GpuMat &last_nmap,
    cv::cuda::GpuMat &sum,
    cv::cuda::GpuMat &out,
    const Sophus::SE3d &pose,
    const Eigen::Matrix3f K,
    float *jtj, float *jtr,
    float *residual)
{
    int cols = curr_vmap.cols;
    int rows = curr_vmap.rows;

    ICPReduction icp;
    icp.out = sum;      // of size 96x29, with one block per row
    icp.curr_vmap_ = curr_vmap;
    icp.curr_nmap_ = curr_nmap;
    icp.last_vmap_ = last_vmap;
    icp.last_nmap_ = last_nmap;
    icp.cols = cols;
    icp.rows = rows;
    icp.N = cols * rows;
    icp.pose = pose.cast<float>().matrix3x4();
    icp.angleThresh = cos(30 * 3.14 / 180);
    icp.distThresh = 0.01;
    icp.fx = K(0,0);
    icp.fy = K(1,1);
    icp.cx = K(0,2);
    icp.cy = K(1,2);

    icp_reduce_kernel<<<96, 224>>>(icp);
    // reduce from sum(96x29) to out(1x29) by sum over all rows in a col
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];   // cost^2
    residual[1] = host_data.ptr<float>()[28];   // 1 for valid cost, 0 for invalid cost
}

} // namespace fusion