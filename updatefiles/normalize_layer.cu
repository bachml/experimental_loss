#include <algorithm>
#include <vector>

#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* center_data = temp_center.mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  //step1 : 计算blob  Batch_num x dim 的向量
  for(int i = 0; i < M_; i++) {
    Dtype dot;
    caffe_gpu_dot(K_, bottom_data + K_*i, bottom_data + K_*i, &dot);
    //temp_center[i] = dot;
    caffe_gpu_set(K_, dot, center_data + K_*i);
    caffe_gpu_powx(K_, center_data + K_*i, (Dtype)0.5, center_data + K_*i);
  }
  //step2: 计算forward 
  caffe_gpu_div(M_ * K_, top_data, center_data, top_data);
  
}


template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* center_data = temp_center.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  // for f_d. f_d' = (1 - f_d^2) / ||f||
  //caffe_set(dim*batch_num, (Dtype)1, bottom_diff)
  caffe_gpu_powx(M_*K_, top_data, (Dtype)2, bottom_diff);
  caffe_gpu_add_scalar(M_*K_, (Dtype)-1, bottom_diff);
  caffe_gpu_scal(M_*K_, (Dtype)-1, bottom_diff);
  caffe_gpu_div(M_*K_, bottom_diff, center_data, bottom_diff);
  caffe_gpu_mul(M_*K_, bottom_diff, top_diff, bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);

}  // namespace caffe
