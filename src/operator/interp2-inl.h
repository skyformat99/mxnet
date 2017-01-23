
#ifndef MXNET_OPERATOR_INTERP2_INL_H_
#define MXNET_OPERATOR_INTERP2_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace interp2_enum {
enum Interp2OpInputs {kData};
enum Interp2OpOutputs {kOut};
}

struct Interp2Param : public dmlc::Parameter<Interp2Param> {
  int height;
  int width;
  int zoom_factor;
  int shrink_factor;
  int pad_beg;
  int pad_end;
  DMLC_DECLARE_PARAMETER(Interp2Param) {
    DMLC_DECLARE_FIELD(height).set_default(0)
    .describe("Heigh of output");

    DMLC_DECLARE_FIELD(width).set_default(0)
    .describe("Width of output");

    DMLC_DECLARE_FIELD(zoom_factor).set_default(1)
    .describe("Zoom factor");

    DMLC_DECLARE_FIELD(shrink_factor).set_default(1)
    .describe("Shrink factor");

    DMLC_DECLARE_FIELD(pad_beg).set_default(0)
    .describe("Padding at begin of input");

    DMLC_DECLARE_FIELD(pad_end).set_default(0)
    .describe("Padding at end of input");
  }
};


template <typename Dtype, bool packed>
void _interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
          Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);
template <typename Dtype, bool packed>
void _interp2_backward(const int channels,
          Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

template<typename xpu, typename DType>
class Interp2Op : public Operator {
  public: 
    explicit Interp2Op(Interp2Param p) {
      this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      TShape dshape = in_data[0].shape_;
      CHECK_GE(dshape.ndim(), 4) << "Interp2: Input data should be 4D in NCHW";
      int num_ = dshape[0];
      int channels_  = dshape[1];
      int height_in_ = dshape[2];
      int width_in_  = dshape[3];
      int height_in_eff_ = height_in_ + param_.pad_beg + param_.pad_end;
      int width_in_eff_  = width_in_ + param_.pad_beg + param_.pad_end;
      int height_out_ = out_data[0].shape_[2];
      int width_out_ = out_data[0].shape_[3];

      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 1);
      _interp2<DType, false>(num_ * channels_,
          (DType *)(in_data[0].dptr_), -param_.pad_beg, -param_.pad_end, height_in_eff_, width_in_eff_, height_in_, width_in_,
          (DType *)(out_data[0].dptr_), 0, 0, height_out_, width_out_, height_out_, width_out_);
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_args) {
      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 1);
      CHECK_EQ(in_grad.size(), 1);
      CHECK_EQ(out_grad.size(), 1);

      TShape dshape = in_data[0].shape_;
      CHECK_GE(dshape.ndim(), 4) << "Interp2: Input data should be 4D in NCHW";
      int num_ = dshape[0];
      int channels_  = dshape[1];
      int height_in_ = dshape[2];
      int width_in_  = dshape[3];
      int height_in_eff_ = height_in_ + param_.pad_beg + param_.pad_end;
      int width_in_eff_  = width_in_ + param_.pad_beg + param_.pad_end;
      int height_out_ = out_data[0].shape_[2];
      int width_out_ = out_data[0].shape_[3];

      _interp2_backward<DType, false>(num_ * channels_,
          (DType *)(in_grad[0].dptr_), -param_.pad_beg, -param_.pad_end, height_in_eff_, width_in_eff_, height_in_, width_in_,
          (DType *)(out_grad[0].dptr_), 0, 0, height_out_, width_out_, height_out_, width_out_);
    }

  private:

    Interp2Param param_;
};

template<typename xpu>
Operator* CreateOp(Interp2Param param, int dtype);

#if DMLC_USE_CXX11
class Interp2Prop : public OperatorProperty {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
      param_.Init(kwargs);
    }

    std::map<std::string, std::string> GetParams() const override {
      return param_.__DICT__();
    }

    bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
      CHECK_EQ(in_shape->size(), 1);
      const TShape &dshape = (*in_shape)[0];
      CHECK_GE(dshape.ndim(), 4) << "Interp2: Input data should be 4D in NCHW";
      int num_ = dshape[0];
      int channels_  = dshape[1];
      int height_in_ = dshape[2];
      int width_in_  = dshape[3];
      int height_in_eff_ = height_in_ + param_.pad_beg + param_.pad_end;
      int width_in_eff_  = width_in_ + param_.pad_beg + param_.pad_end;
      CHECK_GT(height_in_eff_, 0) << "Input height should be positive";
      CHECK_GT(width_in_eff_, 0) << "Input width should be positive";

      TShape oshape = dshape;

      CHECK_GE(param_.shrink_factor, 1) << "Shrink factor must be positive";
      CHECK_GE(param_.zoom_factor, 1) << "Zoom factor must be positive";
      if (param_.shrink_factor != 1 && param_.zoom_factor == 1) {
        oshape[2] = (height_in_eff_ - 1) / param_.shrink_factor + 1;
        oshape[3] = (width_in_eff_ - 1) / param_.shrink_factor + 1;
      } else if (param_.zoom_factor != 1 && param_.shrink_factor == 1) {
        oshape[2] = height_in_eff_ + (height_in_eff_ - 1) * (param_.zoom_factor - 1);
        oshape[3] = width_in_eff_ + (width_in_eff_ - 1) * (param_.zoom_factor - 1);
      } else if (param_.zoom_factor != 1 && param_.shrink_factor != 1) {
        oshape[2] = (height_in_eff_ - 1) / param_.shrink_factor + 1;
        oshape[3] = (width_in_eff_ - 1) / param_.shrink_factor + 1;
        oshape[2] = oshape[2] + (oshape[2] - 1) * (param_.zoom_factor - 1);
        oshape[3] = oshape[3] + (oshape[3] - 1) * (param_.zoom_factor - 1);
      } else {
        LOG(FATAL) << "Unsupport zoom or shrink factor";
      }
      CHECK_GT(oshape[2], 0) << "Output height should be positive";
      CHECK_GT(oshape[3], 0) << "Output width should be positive";
      out_shape->clear();
      out_shape->push_back(oshape);
      return true;
    }
    
    bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      CHECK_EQ(in_type->size(), 1);
      int dtype = (*in_type)[0];
      if (dtype == -1) {
        LOG(FATAL) << "Input type to interp2 is not specified.";
        return false;
      }

      out_type->clear();
      out_type->push_back(dtype);
      return true;
    }

    OperatorProperty* Copy() const override {
      Interp2Prop *prop_sym = new Interp2Prop();
      prop_sym->param_ = this->param_;
      return prop_sym;
    }

    std::string TypeString() const override {
      return "Interp2";
    }

    std::vector<int> DeclareBackwardDependency(const std::vector<int> &out_grad,
                                               const std::vector<int> &in_data,
                                               const std::vector<int> &out_data) const override {
      return {out_grad[interp2_enum::kOut], in_data[interp2_enum::kData], out_data[interp2_enum::kOut]};
    }

    std::vector<std::pair<int, void*> > BackwardInplaceOption(const std::vector<int> &out_grad,
                                                              const std::vector<int> &in_data,
                                                              const std::vector<int> &out_data,
                                                              const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
      return {};
#else
      return {{in_data[interp2_enum::kData], in_grad[interp2_enum::kData]}};
#endif
    }

    Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
    }
  
    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                               std::vector<int> *in_type) const override;

  private:
    Interp2Param param_;
}; // class Interp2Prop

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif

