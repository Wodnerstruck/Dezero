import dezero
from dezero.core import Function, Variable, as_varible, as_array
import numpy as np

class Square(Function):
    def forward(self, x:Variable):
        return x ** 2
    
    def backward(self, gy:Variable):
        x, = self.inputs
        gx = gy * 2 * x
        return gx        

def square(x:Variable):
    return Square()(x)

class Exp(Function):
    def forward(self, x:Variable):
        return np.exp(x)
    
    def backward(self, gy:Variable):
        x, = self.inputs
        gx = np.exp(x) * gy
        return gx

def exp(x:Variable):
    return Exp()(x)

class Sin(Function):
    def forward(self, x:Variable):
        y = np.sin(x)
        return y
    
    def backward(self, gy:Variable):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x:Variable):
    return Sin()(x)

class Cos(Function):
    def forward(self, x:Variable):
        y = np.cos(x)
        return y

    def backward(self, gy:Variable):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x:Variable):
    return Cos()(x)
    
class Tanh(Function):
    def forward(self, x:Variable):
        y = np.tanh(x)
        return y
    
    def backward(self, gy:Variable):
        y = self.outputs[0]() #弱引用需要加括号得到对象本身
        gx = gy * (1 - y * y)
        return gx

def tanh(x:Variable):
    return Tanh()(x)
    
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x: Variable):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy:Variable):
        return reshape(gy, self.x_shape)

def reshape(x:Variable, shape):
    if x.shape == shape:
        return as_varible(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
        
    def backward(self, gy:Variable):
        gx = transpose(gy)
        return gx

def transpose(x:Variable):
    return Transpose()(x)   

from dezero import utils

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x:Variable):
    return Sum()(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_varible(x)
    return BroadcastTo(shape)(x)    

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_varible(x)
    return SumTo(shape)(x)
    