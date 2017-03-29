#include <algorithm>
#include <vector>

#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.normalize_param().axis());
}


template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  M_ = bottom[0]->count(0, axis);
  K_ = bottom[0]->count(axis);


  
  //this->blobs_.resize(1);
  vector<int> blob_shape(2);
  blob_shape[0] = M_;
  blob_shape[1] = K_;

  temp_center.ReshapeLike(*bottom[0]);

  //temp_center.reset(new Blob<Dtype>(blob_shape));

  //this->blobs_[0].reset(new Blob<Dtype>(blob_shape));
  //temp.reset(new Blob<Dtype>(blob_shape));
  //temp.ReshapeLike(*this->blobs_[0]);
}


template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //Dtype* center_data = this->blobs_[0]->mutable_cpu_data();
  Dtype* center_data = temp_center.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  //step1 : 计算blob  Batch_num x dim 的向量
  for(int i = 0; i < M_; i++) {
    Dtype dot = caffe_cpu_dot(K_, bottom_data + K_*i, bottom_data + K_*i);
    //center_data[i] = dot;
    caffe_set(K_, dot, center_data + K_*i);
    caffe_powx(K_, center_data + K_*i, (Dtype)0.5, center_data + K_*i);
  }
  //step2: 计算forward 
  caffe_div(M_ * K_, top_data, center_data, top_data);
}


template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* center_data = temp_center.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  // for f_d. f_d' = (1 - f_d^2) / ||f||
  //caffe_set(dim*batch_num, (Dtype)1, bottom_diff)
  caffe_sqr(M_*K_, top_data, bottom_diff);
  caffe_add_scalar(M_*K_, (Dtype)-1, bottom_diff);
  caffe_scal(M_*K_, (Dtype)-1, bottom_diff);
  caffe_div(M_*K_, bottom_diff, center_data, bottom_diff);
  caffe_mul(M_*K_, bottom_diff, top_diff, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
