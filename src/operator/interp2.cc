
#include "./interp2-inl.h"

namespace mxnet {
namespace op {

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
template <typename Dtype, bool packed>
void _interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
	      const int w1 = w2;
	      if (packed) {
	        const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	        Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	        for (int c = 0; c < channels; ++c) {
	          pos2[0] = pos1[0];
	          pos1++;
	          pos2++;
	        }
	      }
	      else {
	        const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	        Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	        for (int c = 0; c < channels; ++c) {
	          pos2[0] = pos1[0];
	          pos1 += Width1 * Height1;
	          pos2 += Width2 * Height2;
	        }
	      }
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const Dtype w1lambda = w1r - w1;
      const Dtype w0lambda = Dtype(1.) - w1lambda;
      if (packed) {
	      const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	      Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	      for (int c = 0; c < channels; ++c) {
	        pos2[0] =
	          h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[channels * w1p]) + 
	          h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
	        pos1++;
	        pos2++;
	      }
      } else {
	      const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	      Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	      for (int c = 0; c < channels; ++c) {
          // FIXME: fix pos2[0] for gpu segmentation fault
	        pos2[0] =
	          h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
	          h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
	        pos1 += Width1 * Height1;
	        pos2 += Width2 * Height2;
	      }
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, bool packed>
void _interp2_backward(const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  // special case: same-size matching grids
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
	      const int w1 = w2;
	      if (packed) {
	        Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	        const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	        for (int c = 0; c < channels; ++c) {
	          pos1[0] += pos2[0];
	          pos1++;
	          pos2++;
	        }
	      } else {
	        Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	        const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	        for (int c = 0; c < channels; ++c) {
	          pos1[0] += pos2[0];
	          pos1 += Width1 * Height1;
	          pos2 += Width2 * Height2;
	        }
	      }
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const Dtype w1lambda = w1r - w1;
      const Dtype w0lambda = Dtype(1.) - w1lambda;
      if (packed) {
	      Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	      const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	      for (int c = 0; c < channels; ++c) {
	        pos1[0] += h0lambda * w0lambda * pos2[0];
	        pos1[channels * w1p] += h0lambda * w1lambda * pos2[0];
	        pos1[channels * h1p * Width1] += h1lambda * w0lambda * pos2[0];
	        pos1[channels * (h1p * Width1 + w1p)] += h1lambda * w1lambda * pos2[0];
	        pos1++;
	        pos2++;
	      }
      } else {
	      Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	      const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	      for (int c = 0; c < channels; ++c) {
	        pos1[0] += h0lambda * w0lambda * pos2[0];
	        pos1[w1p] += h0lambda * w1lambda * pos2[0];
	        pos1[h1p * Width1] += h1lambda * w0lambda * pos2[0];
	        pos1[h1p * Width1 + w1p] += h1lambda * w1lambda * pos2[0];
	        pos1 += Width1 * Height1;
	        pos2 += Width2 * Height2;
	      }
      }
    }
  }
}

// Explicit instances
template void _interp2<float,false>(const int, const float *, const int, const int, const int, 
    const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void _interp2<float,true>(const int, const float *, const int, const int, const int, 
    const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void _interp2<double,false>(const int, const double *, const int, const int, const int, 
    const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);
template void _interp2<double,true>(const int, const double *, const int, const int, const int, 
    const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);

template void _interp2_backward<float,false>(const int, float *, const int, const int, const int, 
    const int, const int, const int, const float *, const int, const int, const int, const int, const int, const int);
template void _interp2_backward<double,false>(const int, double *, const int, const int, const int, 
    const int, const int, const int, const double *, const int, const int, const int, const int, const int, const int);


template<>
Operator *CreateOp<cpu>(Interp2Param param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {  
    op = new Interp2Op<cpu, DType>(param);
  })

  return op;
}

Operator* Interp2Prop::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(Interp2Param);

MXNET_REGISTER_OP_PROPERTY(Interp2, Interp2Prop)
.describe("Perform 2D interp")
.add_argument("data", "Symbol", "Input data to the interp2 operator")
.add_arguments(Interp2Param::__FIELDS__());

}  // namespace np
}  // namespace mxnet
