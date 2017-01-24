
from skimage import io
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

# IM_PATH = ''
IM_PATH = "/home/hzxiahouzuoxin/zxwkspace/datasets/voc2007/VOCdevkit/VOC2007/JPEGImages/004534.jpg"
# ctx = mx.gpu()
ctx = mx.cpu()

if len(IM_PATH) > 0:
  im = io.imread(IM_PATH)
else:
  im = np.array([[1,5,5,1], [5,1,1,5], [1,5,5,1], [5,1,1,5]])
  im = np.array([im,im,im])

_in = im
_in = np.swapaxes(_in, 0, 2)
_in = np.swapaxes(_in, 1, 2)
print "in_shape: ", _in.shape
_in = _in[np.newaxis,:]

x = mx.sym.Variable('data')
# model = mx.sym.Pooling(data=x, pool_type='max', kernel=(3,3), stride=(2,2))
model = mx.sym.Interp2(data=x, zoom_factor=2)

exc = model.simple_bind(data=_in.shape, ctx=ctx)
exc.forward(data=_in, is_train=False)

_out = mx.nd.zeros(exc.outputs[0].shape)
_out[:] = exc.outputs[0]
_out = _out.asnumpy().astype('uint8')[0]
print "out_shape: ", _out.shape

_out = np.swapaxes(_out, 0, 2)
_out = np.swapaxes(_out, 0, 1)

if len(IM_PATH) > 0:
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
  ax1.imshow(im)
  ax2.imshow(_out)
  plt.show()
else:
  print im
  print _out

