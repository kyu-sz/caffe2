#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "psroi_pool_op.h"

namespace caffe2 {

namespace {

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void PSROIPoolForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int output_dim,
    const int group_size,
    const T* bottom_rois,
    T* top_data,
    int* mapping_channel_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed (too small) RoIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Compute w and h at bottom
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int group_w = pw;
    int group_h = ph;
    int c = (ctop * group_size + group_h) * group_size + group_w;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    T out_sum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        out_sum += bottom_data[bottom_index];
      }
    }

    int bin_area = (hend - hstart) * (wend - wstart);
    top_data[index] = is_empty ? 0. : out_sum / bin_area;
    mapping_channel_ptr[index] = c;
  }
}

template <typename T>
__global__ void PSROIPoolBackward(
    const int nthreads,
    const T* top_diff,
    const int* mapping_channel_ptr,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int output_dim,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / output_dim;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed (too small) RoIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Compute w and h at bottom
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Computer c at bottom
    int c = mapping_channel_ptr[index];
    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    int bin_area = (hend - hstart) * (wend - wstart);
    T diff_val = is_empty ? 0. : top_diff[index] / bin_area;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
      }
    }
  }
}

} // namespace

template <>
bool PSRoIPoolOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto* Y = Output(0); // Position-sensitive RoI pooled data
  auto* MC = is_test_ ? nullptr : Output(1); // Mapping channel

  // Handle empty rois
  if (R.size() == 0) {
    Y->Resize(0, X.dim32(1), pooled_height_, pooled_width_);
    // mutable_data calls are needed to allocate the tensors
    Y->mutable_data<float>();
    if (!is_test_) {
      MC->Resize(Y->dims());
      MC->mutable_data<int>();
    }
    return true;
  }

  Y->Resize(R.dim32(0), X.dim32(1), pooled_height_, pooled_width_);
  if (!is_test_) {
    MC->ResizeLike(Y);
  }
  int output_size = Y->size();
  int* mapping_channel_ptr = is_test_ ? nullptr : MC->mutable_data<int>();

  int count = Y->size();
  math::Set<float, CUDAContext>(count, 0.f, Y->mutable_data<float>(), &context_);
  math::Set<int, CUDAContext>(count, -1, mapping_channel_ptr, &context_);

  PSROIPoolForward<float><<<
      CAFFE_GET_BLOCKS(output_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      output_size,
      X.data<float>(),
      spatial_scale_,
      X.dim32(1),
      X.dim32(2),
      X.dim32(3),
      pooled_height_,
      pooled_width_,
      output_dim_,
      group_size_,
      R.data<float>(),
      Y->mutable_data<float>(),
      mapping_channel_ptr);
  CUDA_POST_KERNEL_CHECK;
  return true;
}

template <>
bool PSRoIPoolGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto& MC = Input(2); // Mapping channel
  auto& dY = Input(3); // Gradient of net w.r.t. output of "forward" op
  // (aka "gradOutput")
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
  // (aka "gradInput")

  dX->ResizeLike(X);
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);
  if (dY.size() > 0) { // Handle possibly empty gradient if there were no rois
    PSROIPoolBackward<float><<<
        CAFFE_GET_BLOCKS(dY.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        dY.size(),
        dY.data<float>(),
        MC.data<int>(),
        R.dim32(0),
        spatial_scale_,
        X.dim32(1),
        X.dim32(2),
        X.dim32(3),
        pooled_height_,
        pooled_width_,
        output_dim_,
        dX->mutable_data<float>(),
        R.data<float>());
  }
  CUDA_POST_KERNEL_CHECK;
  return true;
}

namespace {

REGISTER_CUDA_OPERATOR(RoIPool, RoIPoolOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RoIPoolGradient, RoIPoolGradientOp<float, CUDAContext>);

} // namespace
} // namespace caffe2
