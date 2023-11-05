TYPE_NIL = 0
TYPE_NUMBER = 1
TYPE_STRING = 2
TYPE_TABLE = 3
TYPE_TORCH = 4
TYPE_BOOLEAN = 5
TYPE_FUNCTION = 6
TYPE_RECUR_FUNCTION = 8
LEGACY_TYPE_RECUR_FUNCTION = 7

import struct
from array import array
import numpy as np
import sys
from collections import namedtuple

LuaFunction = namedtuple('LuaFunction',
                         ['size', 'dumped', 'upvalues'])


class hashable_uniq_dict(dict):
    def __hash__(self):
        return id(self)

    def __getattr__(self, key):
        return self.get(key)

    def __eq__(self, other):
        return id(self) == id(other)
    # TODO: dict's __lt__ etc. still exist

torch_readers = {}


def add_tensor_reader(typename, dtype):
    def read_tensor_generic(reader, version):
        ndim = reader.read_int()

        # read size:
        size = reader.read_long_array(ndim)
        # read stride:
        stride = reader.read_long_array(ndim)
        # storage offset:
        storage_offset = reader.read_long() - 1
        # read storage:
        storage = reader.read_obj()

        if storage is None or ndim == 0 or len(size) == 0 or len(stride) == 0:
            # empty torch tensor
            return np.empty((0), dtype=dtype)

        # convert stride to numpy style (i.e. in bytes)
        stride = [storage.dtype.itemsize * x for x in stride]

        # create numpy array that indexes into the storage:
        return np.lib.stride_tricks.as_strided(
            storage[storage_offset:],
            shape=size,
            strides=stride)
    torch_readers[typename] = read_tensor_generic
add_tensor_reader(b'torch.ByteTensor', dtype=np.uint8)
add_tensor_reader(b'torch.CharTensor', dtype=np.int8)
add_tensor_reader(b'torch.ShortTensor', dtype=np.int16)
add_tensor_reader(b'torch.IntTensor', dtype=np.int32)
add_tensor_reader(b'torch.LongTensor', dtype=np.int64)
add_tensor_reader(b'torch.FloatTensor', dtype=np.float32)
add_tensor_reader(b'torch.DoubleTensor', dtype=np.float64)
add_tensor_reader(b'torch.CudaTensor', np.float32)  # float
add_tensor_reader(b'torch.CudaByteTensor', dtype=np.uint8)
add_tensor_reader(b'torch.CudaCharTensor', dtype=np.int8)
add_tensor_reader(b'torch.CudaShortTensor', dtype=np.int16)
add_tensor_reader(b'torch.CudaIntTensor', dtype=np.int32)
add_tensor_reader(b'torch.CudaDoubleTensor', dtype=np.float64)


def add_storage_reader(typename, dtype):
    def read_storage(reader, version):
        size = reader.read_long()
        return np.fromfile(reader.f, dtype=dtype, count=size)
    torch_readers[typename] = read_storage
add_storage_reader(b'torch.ByteStorage', dtype=np.uint8)
add_storage_reader(b'torch.CharStorage', dtype=np.int8)
add_storage_reader(b'torch.ShortStorage', dtype=np.int16)
add_storage_reader(b'torch.IntStorage', dtype=np.int32)
add_storage_reader(b'torch.LongStorage', dtype=np.int64)
add_storage_reader(b'torch.FloatStorage', dtype=np.float32)
add_storage_reader(b'torch.DoubleStorage', dtype=np.float64)
add_storage_reader(b'torch.CudaStorage', dtype=np.float32)  # float
add_storage_reader(b'torch.CudaByteStorage', dtype=np.uint8)
add_storage_reader(b'torch.CudaCharStorage', dtype=np.int8)
add_storage_reader(b'torch.CudaShortStorage', dtype=np.int16)
add_storage_reader(b'torch.CudaIntStorage', dtype=np.int32)
add_storage_reader(b'torch.CudaDoubleStorage', dtype=np.float64)


class TorchObject(object):
    def __init__(self, typename, obj):
        self._typename = typename
        self._obj = obj

    def __getattr__(self, k):
        return self._obj.get(k)

    def __getitem__(self, k):
        return self._obj.get(k)

    def torch_typename(self):
        return self._typename

    def __repr__(self):
        return "TorchObject(%s, %s)" % (self._typename, repr(self._obj))

    def __str__(self):
        return repr(self)

    def __dir__(self):
        keys = list(self._obj.keys())
        keys.append('torch_typename')
        return keys


def add_trivial_class_reader(typename):
    def reader(reader, version):
        obj = reader.read_obj()
        return TorchObject(typename, obj)
    torch_readers[typename] = reader
for mod in [b"nn.ConcatTable", b"nn.SpatialAveragePooling",
            b"nn.TemporalConvolutionFB", b"nn.BCECriterion", b"nn.Reshape", b"nn.gModule",
            b"nn.SparseLinear", b"nn.WeightedLookupTable", b"nn.CAddTable",
            b"nn.TemporalConvolution", b"nn.PairwiseDistance", b"nn.WeightedMSECriterion",
            b"nn.SmoothL1Criterion", b"nn.TemporalSubSampling", b"nn.TanhShrink",
            b"nn.MixtureTable", b"nn.Mul", b"nn.LogSoftMax", b"nn.Min", b"nn.Exp", b"nn.Add",
            b"nn.BatchNormalization", b"nn.AbsCriterion", b"nn.MultiCriterion",
            b"nn.LookupTableGPU", b"nn.Max", b"nn.MulConstant", b"nn.NarrowTable", b"nn.View",
            b"nn.ClassNLLCriterionWithUNK", b"nn.VolumetricConvolution",
            b"nn.SpatialSubSampling", b"nn.HardTanh", b"nn.DistKLDivCriterion",
            b"nn.SplitTable", b"nn.DotProduct", b"nn.HingeEmbeddingCriterion",
            b"nn.SpatialBatchNormalization", b"nn.DepthConcat", b"nn.Sigmoid",
            b"nn.SpatialAdaptiveMaxPooling", b"nn.Parallel", b"nn.SoftShrink",
            b"nn.SpatialSubtractiveNormalization", b"nn.TrueNLLCriterion", b"nn.Log",
            b"nn.SpatialDropout", b"nn.LeakyReLU", b"nn.VolumetricMaxPooling",
            b"nn.KMaxPooling", b"nn.Linear", b"nn.Euclidean", b"nn.CriterionTable",
            b"nn.SpatialMaxPooling", b"nn.TemporalKMaxPooling", b"nn.MultiMarginCriterion",
            b"nn.ELU", b"nn.CSubTable", b"nn.MultiLabelMarginCriterion", b"nn.Copy",
            b"nn.CuBLASWrapper", b"nn.L1HingeEmbeddingCriterion",
            b"nn.VolumetricAveragePooling", b"nn.StochasticGradient",
            b"nn.SpatialContrastiveNormalization", b"nn.CosineEmbeddingCriterion",
            b"nn.CachingLookupTable", b"nn.FeatureLPPooling", b"nn.Padding", b"nn.Container",
            b"nn.MarginRankingCriterion", b"nn.Module", b"nn.ParallelCriterion",
            b"nn.DataParallelTable", b"nn.Concat", b"nn.CrossEntropyCriterion",
            b"nn.LookupTable", b"nn.SpatialSoftMax", b"nn.HardShrink", b"nn.Abs", b"nn.SoftMin",
            b"nn.WeightedEuclidean", b"nn.Replicate", b"nn.DataParallel",
            b"nn.OneBitQuantization", b"nn.OneBitDataParallel", b"nn.AddConstant", b"nn.L1Cost",
            b"nn.HSM", b"nn.PReLU", b"nn.JoinTable", b"nn.ClassNLLCriterion", b"nn.CMul",
            b"nn.CosineDistance", b"nn.Index", b"nn.Mean", b"nn.FFTWrapper", b"nn.Dropout",
            b"nn.SpatialConvolutionCuFFT", b"nn.SoftPlus", b"nn.AbstractParallel",
            b"nn.SequentialCriterion", b"nn.LocallyConnected",
            b"nn.SpatialDivisiveNormalization", b"nn.L1Penalty", b"nn.Threshold", b"nn.Power",
            b"nn.Sqrt", b"nn.MM", b"nn.GroupKMaxPooling", b"nn.CrossMapNormalization",
            b"nn.ReLU", b"nn.ClassHierarchicalNLLCriterion", b"nn.Optim", b"nn.SoftMax",
            b"nn.SpatialConvolutionMM", b"nn.Cosine", b"nn.Clamp", b"nn.CMulTable",
            b"nn.LogSigmoid", b"nn.LinearNB", b"nn.TemporalMaxPooling", b"nn.MSECriterion",
            b"nn.Sum", b"nn.SoftSign", b"nn.Normalize", b"nn.ParallelTable", b"nn.FlattenTable",
            b"nn.CDivTable", b"nn.Tanh", b"nn.ModuleFromCriterion", b"nn.Square", b"nn.Select",
            b"nn.GradientReversal", b"nn.SpatialFullConvolutionMap", b"nn.SpatialConvolution",
            b"nn.Criterion", b"nn.SpatialConvolutionMap", b"nn.SpatialLPPooling",
            b"nn.Sequential", b"nn.Transpose", b"nn.SpatialUpSamplingNearest",
            b"nn.SpatialFullConvolution", b"nn.ModelParallel", b"nn.RReLU",
            b"nn.SpatialZeroPadding", b"nn.Identity", b"nn.Narrow", b"nn.MarginCriterion",
            b"nn.SelectTable", b"nn.VolumetricFullConvolution",
            b"nn.SpatialFractionalMaxPooling", b"fbnn.ProjectiveGradientNormalization",
            b"fbnn.Probe", b"fbnn.SparseLinear", b"cudnn._Pooling3D",
            b"cudnn.VolumetricMaxPooling", b"cudnn.SpatialCrossEntropyCriterion",
            b"cudnn.VolumetricConvolution", b"cudnn.SpatialAveragePooling", b"cudnn.Tanh",
            b"cudnn.LogSoftMax", b"cudnn.SpatialConvolution", b"cudnn._Pooling",
            b"cudnn.SpatialMaxPooling", b"cudnn.ReLU", b"cudnn.SpatialCrossMapLRN",
            b"cudnn.SoftMax", b"cudnn._Pointwise", b"cudnn.SpatialSoftMax", b"cudnn.Sigmoid",
            b"cudnn.SpatialLogSoftMax", b"cudnn.VolumetricAveragePooling", b"nngraph.Node",
            b"nngraph.JustTable", b"graph.Edge", b"graph.Node", b"graph.Graph"]:
	
    add_trivial_class_reader(mod)


class T7ReaderException(Exception):
    pass


class T7Reader:

    def __init__(self,
                 fileobj,
                 use_list_heuristic=True,
                 use_int_heuristic=True,
                 force_deserialize_classes=True,
                 force_8bytes_long=True):

        self.f = fileobj
        self.objects = {}  # read objects so far

        self.use_list_heuristic = use_list_heuristic
        self.use_int_heuristic = use_int_heuristic
        self.force_deserialize_classes = force_deserialize_classes
        self.force_8bytes_long = force_8bytes_long

    def _read(self, fmt):
        sz = struct.calcsize(fmt)
        b = self.f.read(sz)
        if b == b'':
            # print('x')
            s = (0,)
        else:
            s = struct.unpack(fmt, b)

        # print(s)
        return s

    def read_boolean(self):
        return self.read_int() == 1

    def read_int(self):
        return self._read('i')[0]

    def read_long(self):
        if self.force_8bytes_long:
            return self._read('q')[0]
        else:
            return self._read('l')[0]

    def read_long_array(self, n):
        if self.force_8bytes_long:
            lst = []
            for i in range(n):
                lst.append(self.read_long())
            return lst
        else:
            arr = array('l')
            arr.fromfile(self.f, n)
            return arr.tolist()       
    
    def read_float(self):
        return self._read('f')[0]

    def read_double(self):
        return self._read('d')[0]

    def read_string(self):
        size = self.read_int()
        return self.f.read(size)

    def read_obj(self):
        typeidx = self.read_int()
        if typeidx == TYPE_NIL:
            return None
        elif typeidx == TYPE_NUMBER:
            x = self.read_double()
            # Extra checking for integral numbers:
            if self.use_int_heuristic and x.is_integer():
                return int(x)
            return x
        elif typeidx == TYPE_BOOLEAN:
            return self.read_boolean()
        elif typeidx == TYPE_STRING:
            return self.read_string()
        elif (typeidx == TYPE_TABLE or typeidx == TYPE_TORCH
                or typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION
                or typeidx == LEGACY_TYPE_RECUR_FUNCTION):
            # read the index
            index = self.read_int()

            # check it is loaded already
            if index in self.objects:
                return self.objects[index]

            # otherwise read it
            if (typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION
                    or typeidx == LEGACY_TYPE_RECUR_FUNCTION):
                size = self.read_int()
                dumped = self.f.read(size)
                upvalues = self.read_obj()
                obj = LuaFunction(size, dumped, upvalues)
                self.objects[index] = obj
                return obj
            elif typeidx == TYPE_TORCH:
                version = self.read_string()
                if version.startswith(b'V '):
                    versionNumber = int(version.partition(b' ')[2])
                    className = self.read_string()
                else:
                    className = version
                    versionNumber = 0  # created before existence of versioning
                # print(className)
                if className not in torch_readers:
                    if not self.force_deserialize_classes:
                        raise T7ReaderException(
                            'unsupported torch class: <%s>' % className)
                    obj = TorchObject(className, self.read_obj())
                else:
                    obj = torch_readers[className](self, version)
                self.objects[index] = obj
                return obj
            else:  # it is a table: returns a custom dict or a list
                size = self.read_int()
                obj = hashable_uniq_dict()  # custom hashable dict, can be a key
                key_sum = 0                # for checking if keys are consecutive
                keys_natural = True        # and also natural numbers 1..n.
                # If so, returns a list with indices converted to 0-indices.
                for i in range(size):
                    k = self.read_obj()
                    v = self.read_obj()
                    obj[k] = v

                    if self.use_list_heuristic:
                        if not isinstance(k, int) or k <= 0:
                            keys_natural = False
                        elif isinstance(k, int):
                            key_sum += k
                if self.use_list_heuristic:
                    # n(n+1)/2 = sum <=> consecutive and natural numbers
                    n = len(obj)
                    if keys_natural and n * (n + 1) == 2 * key_sum:
                        lst = []
                        for i in range(len(obj)):
                            lst.append(obj[i + 1])
                        obj = lst
                self.objects[index] = obj
                return obj
        else:
            raise T7ReaderException("unknown object")


def load(filename, **kwargs):
    with open(filename, 'rb') as f:
        reader = T7Reader(f, **kwargs)
        return reader.read_obj()
