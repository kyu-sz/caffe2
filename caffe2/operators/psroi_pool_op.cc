#include "psroi_pool_op.h"

#include <cfloat>

namespace caffe2 {

using std::max;
using std::min;

template <>
bool PSRoIPoolOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0); // Input data to pool
  const auto& R = Input(1); // RoIs
  auto* Y = Output(0); // PSRoI pooled data
  auto* MC = is_test_ ? nullptr : Output(1); // Mapping channel

  // Each ROI is of the form [batch_index x1 y1 x2 y2]
  CAFFE_ENFORCE_EQ(R.dim32(1), 5);

  // TODO: Handle the storage_order properly to get the NCWH.
  int batch_size = X.dim32(0);
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);
  int num_rois = R.dim32(0);

  Y->Resize(num_rois, channels, pooled_height_, pooled_width_);
  if (!is_test_) {
    MC->Resize(num_rois, channels, pooled_height_, pooled_width_);
  }

  const float* Xdata = X.data<float>();
  const float* rois = R.data<float>();
  float* Ydata = Y->mutable_data<float>();
  int* mapping_channel_ptr = is_test_ ? nullptr : MC->mutable_data<int>();

  int count = Y->size();
  math::Set<float, CPUContext>(count, 0.f, Y->mutable_data<float>(), &context_);
  math::Set<int, CPUContext>(count, -1, mapping_channel_ptr, &context_);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_id = rois[0];
    int roi_start_w = round(rois[1] * spatial_scale_);
    int roi_start_h = round(rois[2] * spatial_scale_);
    int roi_end_w = round(rois[3] * spatial_scale_);
    int roi_end_h = round(rois[4] * spatial_scale_);
    CAFFE_ENFORCE_GE(roi_batch_id, 0);
    CAFFE_ENFORCE_LT(roi_batch_id, batch_size);

    // Force malformed PSROIs to be 1x1
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);

    const float bin_size_h =
        static_cast<float>(roi_height) / static_cast<float>(pooled_height_);
    const float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(pooled_width_);

    for (int c = 0; c < output_dim_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart =
              static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
          int wstart =
              static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
          int hend =
              static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
          int wend =
              static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart + roi_start_h, 0), height);
          hend = min(max(hend + roi_start_h, 0), height);
          wstart = min(max(wstart + roi_start_w, 0), width);
          wend = min(max(wend + roi_start_w, 0), width);
          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;

          const float* batch_data = Xdata + roi_batch_id * X.size_from_dim(1)
              + ((ph * pooled_width_ + pw) * output_dim_ + c) * X.size_from_dim(2);
          float out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int bottom_index = h * width + w;
              out_sum += batch_data[bottom_index];
            }
          }

          int bin_area = (hend - hstart) * (wend - wstart);
          Ydata[pool_index] = is_empty ? 0. : out_sum / bin_area;
          if (!is_test_) {
            mapping_channel_ptr[pool_index] = c;
          }
        }
      }
      // Increment all data pointers by one channel
      Ydata += Y->size_from_dim(2);
      if (!is_test_) {
        mapping_channel_ptr += MC->size_from_dim(2);
      }
    }
    // Increment PSROI data pointer
    rois += R.size_from_dim(1);
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(PSRoIPool, PSRoIPoolOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PSRoIPoolGradient, PSRoIPoolGradientOp<float, CPUContext>);

// Input: X, rois
// Output case #1: Y, mapping_channel (train mode)
// Output case #2: Y                  (test mode)
OPERATOR_SCHEMA(PSRoIPool)
    .NumInputs(2)
    .NumOutputs({1, 2})
    .TensorInferenceFunction(
      [](const OperatorDef& def, const vector<TensorShape>& in) {
        ArgumentHelper helper(def);
        const StorageOrder order = StringToStorageOrder(
            helper.GetSingleArgument<string>("order", "NCHW"));
        const TensorShape &X = in[0];
        const int num_channels =
            (order == StorageOrder::NCHW ? X.dims(1) : X.dims(3));
        const TensorShape &R = in[1];
        const int num_rois = R.dims(0);
        const int group_size = helper.GetSingleArgument<int>("group_size", 1);
        TensorShape Y = CreateTensorShape(
            vector<int>({num_rois, num_channels, group_size, group_size}),
            X.data_type());

        bool is_test = (bool) helper.GetSingleArgument<int>("is_test", 0);
        if (!is_test) {
          TensorShape mapping_channel = Y;
          mapping_channel.set_data_type(TensorProto_DataType_INT32);
          return vector<TensorShape>({Y, mapping_channel});
        } else {
          return vector<TensorShape>({Y});
        }
      })
    .SetDoc(R"DOC(
Carries out Position-sensitive RoI Pooling for R-FCN.
Depending on the mode, there are multiple output cases:

  Output case #1: Y, mapping_channel (train mode)
  Output case #2: Y           (test mode)
)DOC")
    .Arg(
        "is_test",
        "If set, run in test mode and skip computation of mapping_channel (used for "
        "gradient computation). Only one output tensor is produced. "
        "(Default: false).")
    .Arg("order", "MC StorageOrder string (Default: \"NCHW\").")
    .Arg("output_dim", "Output channel number (classification class number + 1).")
    .Arg("group_size", "The pooled output group size (Default: 1).")
    .Arg(
        "spatial_scale",
        "Multiplicative spatial scale factor to translate PSROI coords from "
        "their input scale to the scale used when pooling (Default: 1.0).")
    .Input(
        0,
        "X",
        "The input 4-D tensor of data. Only NCHW order is currently supported.")
    .Input(
        1,
        "rois",
        "RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of "
        "shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].")
    .Output(
        0,
        "Y",
        "Position-sensitive RoI pooled output 4-D tensor of shape "
        "(num_rois, channels, pooled_h, pooled_w).")
    .Output(
        1,
        "mapping_channel",
        "Mapping channel used for gradient computation. "
        "Only output if arg \"is_test\" is false.");

// Input: X, rois, mapping_channel, dY (aka "gradOutput")
// Output: dX (aka "gradInput")
OPERATOR_SCHEMA(PSRoIPoolGradient).NumInputs(4).NumOutputs(1);

class GetPSRoIPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PSRoIPoolGradient",
        "",
        vector<string>{I(0), I(1), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(PSRoIPool, GetPSRoIPoolGradient);

} // namespace
} // namespace caffe2
