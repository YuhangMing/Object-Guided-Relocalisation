#include "math/matrix_type.h"
#include "math/vector_type.h"
#include "tracking/m_estimator.h"
#include "utils/safe_call.h"
#include "tracking/reduce_sum.h"
#include <thrust/device_vector.h>

namespace fusion
{

// TODO : Robust RGB Estimation
// STATUS: On halt
// struct RGBSelection
// {
//     __device__ inline bool find_corresp(
//         const int &x,
//         const int &y,
//         float &curr_val,
//         float &last_val,
//         float &dx,
//         float &dy,
//         Vector4f &pt) const
//     {
//         // reference point
//         pt = last_vmap.ptr(y)[x];
//         if (isnan(pt.x) || pt.w < 0)
//             return false;

//         // reference point in curr frame
//         pt = T_last_curr(pt);

//         // reference intensity
//         last_val = last_intensity.ptr(y)[x];

//         if (!isfinite(last_val))
//             return false;

//         auto u = fx * pt.x / pt.z + cx;
//         auto v = fy * pt.y / pt.z + cy;
//         if (u >= 1 && v >= 1 && u <= cols - 2 && v <= rows - 2)
//         {
//             curr_val = interpolate_bilinear(curr_intensity, u, v);
//             dx = interpolate_bilinear(curr_intensity_dx, u, v);
//             dy = interpolate_bilinear(curr_intensity_dy, u, v);

//             // point selection criteria
//             // TODO : Optimise this
//             return (dx > 2 || dy > 2) &&
//                    isfinite(curr_val) &&
//                    isfinite(dx) && isfinite(dy);
//         }

//         return false;
//     }

//     __device__ float interpolate_bilinear(cv::cuda::PtrStep<float> image, float &x, float &y) const
//     {
//         int u = std::floor(x), v = std::floor(y);
//         float coeff_x = x - u, coeff_y = y - v;
//         return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
//                (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
//     }

//     __device__ __inline__ void operator()() const
//     {
//         for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
//         {
//             const int y = k / cols;
//             const int x = k - y * cols;

//             if (y >= cols || x >= rows)
//                 return;

//             Vector4f pt;
//             float curr_val, last_val, dx, dy;
//             bool corresp_found = find_corresp(x, y, curr_val, last_val, dx, dy, pt);

//             if (corresp_found)
//             {
//                 uint index = atomicAdd(num_corresp, 1);
//                 array_image[index] = Vector4f(last_val, curr_val, dx, dy);
//                 array_point[index] = pt;
//                 error_term[index] = pow(curr_val - last_val, 2);
//             }
//         }
//     }

//     cv::cuda::PtrStep<Vector4f> last_vmap;
//     cv::cuda::PtrStep<float> last_intensity;
//     cv::cuda::PtrStep<float> curr_intensity;
//     cv::cuda::PtrStep<float> curr_intensity_dx;
//     cv::cuda::PtrStep<float> curr_intensity_dy;
//     float fx, fy, cx, cy;
//     DeviceMatrix3x4 T_last_curr;
//     Matrix3x3f RLastCurr;
//     Vector3f TLastCurr;
//     int N, cols, rows;

//     int *num_corresp;
//     Vector4f *array_image;
//     Vector4f *array_point;

//     float *error_term;
// };

// __global__ void compute_rgb_corresp_kernel(RGBSelection delegate)
// {
//     delegate();
// }

// __global__ void compute_variance_kernel(float *error_term, float *variance_term, float mean, uint max_idx)
// {
//     uint x = threadIdx.x + blockDim.x * blockIdx.x;
//     if (x >= max_idx)
//         return;

//     variance_term[x] = pow(error_term[x] - mean, 2);
// }

// void compute_rgb_corresp(
//     const cv::cuda::GpuMat last_vmap,
//     const cv::cuda::GpuMat last_intensity,
//     const cv::cuda::GpuMat curr_intensity,
//     const cv::cuda::GpuMat curr_intensity_dx,
//     const cv::cuda::GpuMat curr_intensity_dy,
//     const Sophus::SE3d &frame_pose,
//     const IntrinsicMatrix K,
//     Vector4f *transformed_points,
//     Vector4f *image_corresp_data,
//     float *error_term_array,
//     float *variance_term_array,
//     float &mean_estimate,
//     float &stdev_estimated,
//     uint &num_corresp)
// {
//     auto cols = last_vmap.cols;
//     auto rows = last_vmap.rows;

//     RGBSelection delegate;
//     delegate.last_vmap = last_vmap;
//     delegate.last_intensity = last_intensity;
//     delegate.curr_intensity = curr_intensity;
//     delegate.curr_intensity_dx = curr_intensity_dx;
//     delegate.curr_intensity_dy = curr_intensity_dy;
//     delegate.T_last_curr = frame_pose;
//     delegate.array_image = image_corresp_data;
//     delegate.array_point = transformed_points;
//     delegate.error_term = error_term_array;
//     delegate.fx = K.fx;
//     delegate.fy = K.fy;
//     delegate.cx = K.cx;
//     delegate.cy = K.cy;
//     delegate.cols = cols;
//     delegate.rows = rows;
//     delegate.N = cols * rows;

//     safe_call(cudaMalloc(&delegate.num_corresp, sizeof(uint)));
//     safe_call(cudaMemset(delegate.num_corresp, 0, sizeof(uint)));

//     compute_rgb_corresp_kernel<<<96, 224>>>(delegate);

//     safe_call(cudaMemcpy(&num_corresp, delegate.num_corresp, sizeof(uint), cudaMemcpyDeviceToHost));

//     if (num_corresp <= 1)
//         return;

//     thrust::device_ptr<float> error_term(error_term_array);
//     thrust::device_ptr<float> variance_term(variance_term_array);

//     float sum_error = thrust::reduce(error_term, error_term + num_corresp);
//     mean_estimate = 0;
//     stdev_estimated = std::sqrt(sum_error / (num_corresp - 6));

//     // dim3 thread(MAX_THREAD);
//     // dim3 block(div_up(num_corresp, thread.x));

//     // compute_variance_kernel<<<block, thread>>>(error_term_array, variance_term_array, mean_estimate, num_corresp);
//     // float sum_variance = thrust::reduce(variance_term, variance_term + num_corresp);
//     // stdev_estimated = sqrt(sum_variance / (num_corresp - 1));

//     std::cout << "mean : " << mean_estimate << " stddev : " << stdev_estimated << " num_corresp : " << num_corresp << std::endl;

//     safe_call(cudaFree(delegate.num_corresp));
// }

// // TODO : Robust RGB Estimation
// // STATUS: On halt
// struct RGBLeastSquares
// {
//     cv::cuda::PtrStep<float> out;

//     Vector4f *transformed_points;
//     Vector4f *image_corresp_data;
//     float mean_estimated;
//     float stdev_estimated;
//     uint num_corresp;
//     float fx, fy;
//     size_t N;

//     __device__ void compute_jacobian(const int &k, float *sum)
//     {
//         float row[7] = {0, 0, 0, 0, 0, 0, 0};
//         float weight = 0;
//         if (k < num_corresp)
//         {
//             Vector3f p_transformed = ToVector3(transformed_points[k]);
//             Vector4f image = image_corresp_data[k];

//             float z_inv = 1.0 / p_transformed.z;
//             Vector3f left;
//             left.x = image.z * fx * z_inv;
//             left.y = image.w * fy * z_inv;
//             left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;

//             float residual = image.y - image.x; // curr_val - last_val
//             float res_normalized = residual / stdev_estimated;
//             float threshold_huber = 1.345 * stdev_estimated;

//             if (fabs(res_normalized) < threshold_huber)
//                 weight = 1;
//             else
//                 weight = threshold_huber / fabs(res_normalized);

//             row[6] = (-residual);
//             // printf("%f, %f\n", res_normalized, threshold_huber);
//             *(Vector3f *)&row[0] = left;
//             *(Vector3f *)&row[3] = p_transformed.cross(left);
//         }

//         int count = 0;
// #pragma unroll
//         for (int i = 0; i < 7; ++i)
//         {
// #pragma unroll
//             for (int j = i; j < 7; ++j)
//                 sum[count++] = row[i] * row[j];
//         }
//     }

//     __device__ void operator()()
//     {
//         float sum[29] = {0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0};

//         float val[29];
//         for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
//         {
//             compute_jacobian(k, val);

// #pragma unroll
//             for (int i = 0; i < 29; ++i)
//                 sum[i] += val[i];
//         }

//         BlockReduce<float, 29>(sum);

//         if (threadIdx.x == 0)
//         {
// #pragma unroll
//             for (int i = 0; i < 29; ++i)
//                 out.ptr(blockIdx.x)[i] = sum[i];
//         }
//     }
// }; // struct RGBLeastSquares

// __global__ void compute_least_square_RGB_kernel(RGBLeastSquares delegate)
// {
//     delegate();
// }

// // TODO : Robust RGB Estimation
// // STATUS: On halt
// void compute_least_square_RGB(
//     const uint num_corresp,
//     Vector4f *transformed_points,
//     Vector4f *image_corresp_data,
//     const float mean_estimated,
//     const float stdev_estimated,
//     const IntrinsicMatrix K,
//     cv::cuda::GpuMat sum,
//     cv::cuda::GpuMat out,
//     float *hessian_estimated,
//     float *residual_estimated,
//     float *residual)
// {
//     RGBLeastSquares delegate;
//     delegate.fx = K.fx;
//     delegate.fy = K.fy;
//     delegate.out = sum;
//     delegate.N = num_corresp;
//     delegate.num_corresp = num_corresp;
//     delegate.image_corresp_data = image_corresp_data;
//     delegate.transformed_points = transformed_points;
//     delegate.mean_estimated = mean_estimated;
//     delegate.stdev_estimated = stdev_estimated;

//     compute_least_square_RGB_kernel<<<96, 224>>>(delegate);
//     cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

//     cv::Mat host_data;
//     out.download(host_data);
//     create_jtjjtr<6, 7>(host_data, hessian_estimated, residual_estimated);
//     residual[0] = host_data.ptr<float>()[27];
//     residual[1] = num_corresp;

//     // std::cout << residual[0] << " : " << residual[1] << std::endl;
// }

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
              const IntrinsicMatrix K,
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
    rr.fx = K.fx;
    rr.fy = K.fy;
    rr.cx = K.cx;
    rr.cy = K.cy;
    rr.invfx = K.invfx;
    rr.invfy = K.invfy;
    rr.out = sum;

    rgb_reduce_kernel2<<<96, 224>>>(rr);
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = host_data.ptr<float>()[28];
}

} // namespace fusion