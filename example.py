## So a function that takes the loss as input
## And my_net.parameters
## And returns the full Hessian?

def hessian(loss, network_param):
  
  
  """
  Calculates the full Hessian of a Neural Network
  Args:
    loss: The calculated loss value
    network_param: A generator containing the parameters of the neural network
  
  
  """
  ## loss should be entry of the form loss = loss_fn(out, y)
  ## network_param should be my_net.parameters()
  
  param_list = [param for param in network_param]
  first_derivative = torch.autograd.grad(loss, param_list, create_graph=True)
  derivative_tensor = torch.cat([tensor.flatten() for tensor in first_derivative])
  num_parameters = derivative_tensor.shape[0]
  hessian = torch.zeros(num_parameters, num_parameters)
  
  for col_ind in range(num_parameters):
    jacobian_vec = torch.zeros(num_parameters)
    jacobian_vec[col_ind] = 1.
    if not col_ind == 0:
      for param in param_list:
        param.grad.zero_()
    #my_net.zero_grad()
    derivative_tensor.backward(jacobian_vec, retain_graph = True)
    hessian_col = torch.cat([param.grad.flatten() for param in param_list])
    hessian[:,col_ind] = hessian_col
  return hessian


import torch
import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1,2)
    self.fc2 = nn.Linear(2,1)
  
  def forward(self, x):
    x = F.softplus(self.fc1(x))
    #x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
  
## I think biases are init to 1 so that is why the last diagonal is always 2
my_net = Network()
x_data = torch.Tensor(2,1).normal_()
y_data = torch.Tensor(2,1).normal_()

opt = torch.optim.SGD(my_net.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

out = my_net(x_data)
loss = loss_fn(out, y_data)

print(hessian(loss, my_net.parameters()))


