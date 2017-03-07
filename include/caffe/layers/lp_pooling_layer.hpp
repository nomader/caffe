#ifndef CAFFE_LP_POOLING_LAYER_HPP_
#define CAFFE_LP_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

/**
 * @brief Layer that performs Lp pooling.
 */
template <typename Dtype>
class LpPoolingLayer : public PoolingLayer<Dtype> {
 public:
  explicit LpPoolingLayer(const LayerParameter& param)
	  : PoolingLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LpPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  bool channel_shared_; //Parameter for sharing single p over all channels  
  bool normalize_input_; //Parameter for normalizing input values of pooling between [-1,1] (x_i/max(|x_i|))
 
 private:
  static bool abs_compare(Dtype a, Dtype b); //Custom comparison function used in finding max abs val
};

}  // namespace caffe

#endif  // CAFFE_LP_POOLING_LAYER_HPP_
