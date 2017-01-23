
#include "./interp2-inl.h"

namespace mxnet {
namespace op {

template <>
Operator *CreateOp<gpu>(Interp2Param param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Interp2Op<gpu, DType>(param);
  })
  return op;
}

}
}
