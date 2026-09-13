class Value:
  def __init__(self, data, children=()):
    self.data= data
    self.prev= set(children)
    self._backward = lambda: None
    self.grad = 0.0

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
 
  def __add__(self, other):
    if not isinstance(other, Value):
      other= Value(other)

    out= Value(self.data + other.data, (self, other))

    def _backward():
      self.grad+= 1 * out.grad
      other.grad+= 1 * out.grad

    out._backward= _backward

    return out

  def __sub__(self, other):
    return self + (-1*other)

  def __mul__(self, other):
    if not isinstance(other, Value):
      other= Value(other)

    out = Value(self.data * other.data, (self, other))

    def _backward():
      self.grad+= other.data*out.grad
      other.grad+= self.data*out.grad

    out._backward = _backward

    return out

  def __pow__(self, other):
    out = Value(self.data**other, (self,))

    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward

    return out

  def relu(self):
    res= max(0, self.data)
    out= Value(res, (self, ))

    def _backward():
      self.grad+= (1 if self.data > 0 else 0) * out.grad

    out._backward= _backward

    return out

  def __neg__(self):
    return self * -1

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return Value(other) + (-self)

  def __rmul__(self, other):
    return self * other
 
  def backward(self)-> None:

    visited= set()
    topo=[]
    def build_topo(value: Value):
      if value not in visited:
        visited.add(value)
        for child in value.prev:
          build_topo(child)
        topo.append(value)
     
    build_topo(self)

    self.grad=1
    for n in reversed(topo):
      n._backward()

 
