import numpy as np
import weakref
import contextlib
import dezero
from typing import Optional
#import dezero.functions


class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    #__array_priority__ = 2000
    def __init__(self, data:np.ndarray, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator: Function = None
        self.generation = 0
    
    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            #self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data)) # 可求高阶导数
            
        funs:list[Function] = []
        seen_set = set()

        def add_func(f: Function):
            if f not in seen_set:
                funs.append(f)
                seen_set.add(f)
                funs.sort(key=lambda x : x.generation) 

        add_func(self.creator)
        while funs:
            f = funs.pop()
            gys = [output().grad for output in f.outputs]
            with using_config('enable_backprop', create_graph): #用于高阶导数的计算图
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx # 相同变量的梯度累加

                    if x.creator is not None:
                        add_func(x.creator)
                
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None #清除中间导数

    def cleargrad(self):
        self.grad = None
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def transpose(self):
        return dezero.functions.transpose(self)
    
    @property
    def T(self):
        return dezero.functions.transpose(self)
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' *  9)
        return 'varible('+ p + ')'
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return neg(self)
    
    def __sub__(self, other):
        return sub(self, other)
    def __rsub__(self, other):
        return rsub(self, other)
    
    def __div__(self, other):
        return div(self, other)
    
    def __rdiv__(self, other):
        return rdiv(self, other)
    
    def __pow__(self, other):
        return pow(self, other)  

class Function:
    def __call__(self, *inputs: Optional[Variable]):
        inputs = [as_varible(x) for x in inputs]
        xs = [x.data for x in inputs] #data是 ndarray类型的
        ys = self.forward(*xs) #Function的 forward返回 ndarray
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] #最后转为Variable

        if Config.enable_backprop: #建立计算图
            self.generation = max(x.generation for x in inputs)

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs] #对输出变量弱引用，打破循环引用提高内存效率
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x:np.ndarray):
        raise NotImplementedError()
    
    def backward(self, gy:Variable):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0:np.ndarray, x1:np.ndarray):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return (y, )
    
    def backward(self, gy:Variable):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0,self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0:np.ndarray, x1:np.ndarray):
        y = x0 * x1
        return y
    
    def backward(self, gy:Variable):
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0,x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x:np.ndarray):
        return -x
    
    def backward(self, gy:Variable):
        return -gy

class Sub(Function):
    def forward(self, x0:np.ndarray, x1:np.ndarray):
        self.x0_shape , self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy:Variable):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0,self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0:np.ndarray, x1:np.ndarray):
        y = x0 / x1
        return y
    
    def backward(self, gy:Variable):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0,x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x:np.ndarray):
        y = x ** self.c
        return y
    
    def backward(self, gy:Variable):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_varible(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)