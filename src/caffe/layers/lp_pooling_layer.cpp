#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>

#include "caffe/layers/lp_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

using namespace std; //Added by AMOGH because of VS14 error: 'max': identifier not found

namespace caffe {

template <typename Dtype>
void LpPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  LpParameter lp_param = this->layer_param_.lp_param();
  this->channels_ = bottom[0]->channels();
  channel_shared_ = lp_param.channel_shared();
  normalize_input_ = lp_param.normalize_input();
  if (this->blobs_.size() > 0) {
	  LOG(INFO) << "Skipping parameter initialization";
  }
  else {
	  this->blobs_.resize(1);
	  if (channel_shared_) {
		  this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
	  }
	  else {
		  this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels_)));
	  }
	  shared_ptr<Filler<Dtype> > p_filler;
	  if (lp_param.has_p_filler()) {
		  p_filler.reset(GetFiller<Dtype>(lp_param.p_filler()));
	  }
	  else {
		  FillerParameter p_filler_param;
		  p_filler_param.set_type("constant");
		  p_filler_param.set_value(2);
		  p_filler.reset(GetFiller<Dtype>(p_filler_param));
	  }
	  p_filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
	  CHECK_EQ(this->blobs_[0]->count(), 1)
		  << "Lp power param size is inconsistent with prototxt config";
  }
  else {
	  CHECK_EQ(this->blobs_[0]->count(), channels_)
		  << "Lp power param size is inconsistent with prototxt config";
  }
  
  if (pool_param.global_pooling()) {
	  CHECK(!(pool_param.has_kernel_size() ||
		  pool_param.has_kernel_h() || pool_param.has_kernel_w()))
		  << "With Global_pooling: true Filter size cannot be specified";
  }
  else {
	  CHECK(!pool_param.has_kernel_size() !=
		  !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
		  << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
	  CHECK(pool_param.has_kernel_size() ||
		  (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
		  << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
	  && pool_param.has_pad_w())
	  || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
	  << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
	  && pool_param.has_stride_w())
	  || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
	  << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
	  this->kernel_h_ = bottom[0]->height();
	  this->kernel_w_ = bottom[0]->width();
  }
  else {
	  if (pool_param.has_kernel_size()) {
		  this->kernel_h_ = this->kernel_w_ = pool_param.kernel_size();
	  }
	  else {
		  this->kernel_h_ = pool_param.kernel_h();
		  this->kernel_w_ = pool_param.kernel_w();
	  }
  }
  CHECK_GT(this->kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(this->kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
	  this->pad_h_ = pad_w_ = pool_param.pad();
  }
  else {
	  this->pad_h_ = pool_param.pad_h();
	  this->pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
	  this->stride_h_ = this->stride_w_ = pool_param.stride();
  }
  else {
	  this->stride_h_ = pool_param.stride_h();
	  this->stride_w_ = pool_param.stride_w();
  }
  if (this->global_pooling_) {
	  CHECK(this->pad_h_ == 0 && this->pad_w_ == 0 && this->stride_h_ == 1 && this->stride_w_ == 1)
		  << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (this->pad_h_ != 0 || this->pad_w_ != 0) {
	  CHECK_LT(this->pad_h_, this->kernel_h_);
	  CHECK_LT(this->pad_w_, this->kernel_w_);
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}
  
template <typename Dtype>
void LpPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	this->channels_ = bottom[0]->channels();
	this->height_ = bottom[0]->height();
	this->width_ = bottom[0]->width();

	//Double check if nr.channels are changed dynamically after initialization
	if (channel_shared_) {
		CHECK_EQ(this->blobs_[0]->count(), 1)
			<< "Lp power param size is inconsistent with prototxt config";
	}
	else {
		CHECK_EQ(this->blobs_[0]->count(), channels_)
			<< "Lp power param size is inconsistent with prototxt config";
	}

	if (this->global_pooling_) {
		this->kernel_h_ = bottom[0]->height();
		this->kernel_w_ = bottom[0]->width();
	}
	this->pooled_height_ = static_cast<int>(ceil(static_cast<float>(
		this->height_ + 2 * this->pad_h_ - this->kernel_h_) / this->stride_h_)) + 1;
	this->pooled_width_ = static_cast<int>(ceil(static_cast<float>(
		this->width_ + 2 * this->pad_w_ - this->kernel_w_) / this->stride_w_)) + 1;
	if (this->pad_h_ || this->pad_w_) {
		// If we have padding, ensure that the last pooling starts strictly
		// inside the image (instead of at the padding); otherwise clip the last.
		if ((this->pooled_height_ - 1) * this->stride_h_ >= this->height_ + this->pad_h_) {
			--this->pooled_height_;
		}
		if ((this->pooled_width_ - 1) * this->stride_w_ >= this->width_ + this->pad_w_) {
			--this->pooled_width_;
		}
		CHECK_LT((this->pooled_height_ - 1) * this->stride_h_, this->height_ + this->pad_h_);
		CHECK_LT((this->pooled_width_ - 1) * this->stride_w_, this->width_ + this->pad_w_);
	}
	top[0]->Reshape(bottom[0]->num(), this->channels_, this->pooled_height_,
		this->pooled_width_);
	if (top.size() > 1) {
		top[1]->ReshapeLike(*top[0]);
	}
}

template <typename Dtype>
void LpPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int top_count = top[0]->count();
	const Dtype* power_data = this->blobs_[0]->cpu_data();
	const int power_count = this->blobs_[0]->count();

	//Set all top data to zero
	for (int i = 0; i < top_count; ++i) {
		top_data[i] = 0;
	}
	// The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < this->channels_; ++c) {
		//Dtype power = 1 + log(1 + exp(power_data[power_count == 1 ? 0 : c])); //Reparameterize p = 1 + log(1 + e^p)
		Dtype power = max(power_data[power_count == 1 ? 0 : c], Dtype(1)); //Reparameterize p = max(p,1)
        for (int ph = 0; ph < this->pooled_height_; ++ph) {
          for (int pw = 0; pw < this->pooled_width_; ++pw) {
            int hstart = ph * this->stride_h_ - this->pad_h_;
			int wstart = pw * this->stride_w_ - this->pad_w_;
			int hend = min(hstart + this->kernel_h_, this->height_ + this->pad_h_);
			int wend = min(wstart + this->kernel_w_, this->width_ + this->pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
			hend = min(hend, this->height_);
			wend = min(wend, this->width_);
			//TODO: caffe_powx(caffe_abs(bottom_data[h * width_ + w]));
			Dtype input_normalizer = *max_element(&bottom_data[hstart * this->width_ + wstart], &bottom_data[hend * this->width_ + wend], abs_compare); //Compute normalization term for input x_n/max(|x_n|)
			//Forward pass //y = ((1/N)*Sum_n(|x_n|^p)^(1/p))
			for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
				Dtype input = bottom_data[h * this->width_ + w]; //nonconst copy of x_n
				//if (input_normalizer == 0) input = Dtype(1); //Hack//Set x_n to 1 if all elements of x are 0
				if (normalize_input_ && input_normalizer != 0) input /= input_normalizer; //Normalize input (for regular cases)
				top_data[ph * this->pooled_width_ + pw] += pow(abs(input), power ); //Sum_n(|x_n|^p))
              }
            }
			top_data[ph * this->pooled_width_ + pw] /= pool_size; // *(1/N)
			top_data[ph * this->pooled_width_ + pw] = pow(top_data[ph * this->pooled_width_ + pw], (1.0/power)); //^(1/p)
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
}

template <typename Dtype>
void LpPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	const int top_count = top[0]->count();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* power_diff = this->blobs_[0]->mutable_cpu_diff();
	const Dtype* power_data = this->blobs_[0]->cpu_data();
	const int power_count = this->blobs_[0]->count();

	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	// The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < this->channels_; ++c) {
		//Dtype power = 1 + log(1 + exp(power_data[power_count == 1 ? 0 : c])); //Reparameterize p = 1 + log(1 + e^p)
		Dtype power = max(power_data[power_count == 1 ? 0 : c], Dtype(1)); //Reparameterize p = max(p,1)
        for (int ph = 0; ph < this->pooled_height_; ++ph) {
          for (int pw = 0; pw < this->pooled_width_; ++pw) {
            int hstart = ph * this->stride_h_ - this->pad_h_;
			int wstart = pw * this->stride_w_ - this->pad_w_;
			int hend = min(hstart + this->kernel_h_, this->height_ + this->pad_h_);
			int wend = min(wstart + this->kernel_w_, this->width_ + this->pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
			hend = min(hend, this->height_);
			wend = min(wend, this->width_);
			Dtype input_normalizer = *max_element(&bottom_data[hstart * this->width_ + wstart], &bottom_data[hend * this->width_ + wend], abs_compare); //Compute normalization term for input x_n/max(|x_n|)
			
			Dtype tmp = Dtype(0); //Temporary variable to accumulate sum //tmp = Sum_n[(|x_n|^p)*ln|x_n|]
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
				Dtype input = bottom_data[h * this->width_ + w]; //nonconst copy of x_n
				//if (input_normalizer == 0) input = Dtype(1); //Hack//Set x_n to 1 if all elements of x are 0
				if (normalize_input_ && input_normalizer != 0) input /= input_normalizer; //Normalize input (for regular cases)
				if (input == 0) continue; //Skip if x_n == 0 [b/c ln|x_n| = -inf]
				//Computing term used for param gradients (power) //Sum_n[(|x_n|^p)*ln|x_n|]
				tmp += pow(abs(input), power) * log(abs(input));
              }
            }

			for (int h = hstart; h < hend; ++h) {
			  for (int w = wstart; w < wend; ++w) {
				if (top_data[ph * this->pooled_width_ + pw] == 0) continue; //Skip if y = 0
				
				Dtype input = bottom_data[h * this->width_ + w]; //nonconst copy of x_n
				//if (input_normalizer == 0) input = Dtype(1); //Hack//Set x_n to 1 if all elements of x are 0
				if (normalize_input_ && input_normalizer != 0) input /= input_normalizer; //Normalize input (for regular cases)
				//Computing gradients wrt bottom data //delta_x_n = delta_y * y^(1-p)/N * x_n/|x_n| * |x_n|^(p-1)

				bottom_diff[h * this->width_ + w] += top_diff[ph * this->pooled_width_ + pw] // delta_y
				  * (pow(top_data[ph * this->pooled_width_ + pw], 1 - power) / pool_size) // * y^(1-p)/N
				  * (input >= 0 ? 1 : -1) // * x_n/|x_n| [if x_n == 0, set x_n/|x_n| = 1]
				  * pow(abs(input), power - 1); // * |x_n|^(p-1)
				
				//Computing gradients wrt power param //delta_p = delta_y * (y / p) * [ tmp/(N*y^p) - ln(y) ] 
				//If p is reparameterized, compute delta_p' = delta_p * dp/dp' = delta_p * e^p'/(1+e^p')
				power_diff[power_count == 1 ? 0 : c] += top_diff[ph * this->pooled_width_ + pw] // delta_y
					* (top_data[ph * this->pooled_width_ + pw] / power) // * (y / p) 
					* (tmp / ( pool_size * pow(top_data[ph * this->pooled_width_ + pw], power)) - log(top_data[ph * this->pooled_width_ + pw])) // * [ tmp/(N*y^p) - ln(y) ]
					//* (exp(power_data[power_count == 1 ? 0 : c]) / (1 + exp(power_data[power_count == 1 ? 0 : c]))); // * e^p'/(1+e^p') [when p = 1 + log(1+e^p')]
					* Dtype(power_data[power_count == 1 ? 0 : c] >= 1); // * 0 if p' < 0 | 1 otherwise [when p = max(p',1)]
			  }
			}
          }
        }
		// offset (to goto the next channel)
		bottom_diff += bottom[0]->offset(0, 1);
		top_diff += top[0]->offset(0, 1);
      }
    }
}

template <typename Dtype>
bool LpPoolingLayer<Dtype>::abs_compare(Dtype a, Dtype b)
{
	return (abs(a) < abs(b));
}


#ifdef CPU_ONLY
STUB_GPU(LpPoolingLayer);
#endif

INSTANTIATE_CLASS(LpPoolingLayer);
REGISTER_LAYER_CLASS(LpPooling);
}  // namespace caffe