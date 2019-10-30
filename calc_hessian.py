def calc_hessian(loss, network_param):
  
  
  """
  Calculates the full Hessian of a Neural Network
  Args:
    loss: The calculated loss value, e.g, loss = loss_fn(my_net(x), y)
    network_param: A generator containing the parameters of the neural network, e.g, network_param = my_net.parameters()
  Returns:
    The full hessian
  The order of the parameters in the Hessian are those that are obtained by creating the list
  [param.flatten() for param in my_net.parameters()]
  """
  
  param_list = [param for param in network_param]
  first_derivative = torch.autograd.grad(loss, param_list, create_graph=True)
  derivative_tensor = torch.cat([tensor.flatten() for tensor in first_derivative])
  num_parameters = derivative_tensor.shape[0]
  hessian = torch.zeros(num_parameters, num_parameters)
  
  for col_ind in range(num_parameters):
    jacobian_vec = torch.zeros(num_parameters)
    jacobian_vec[col_ind] = 1.
    for param in param_list:
        param.grad.zero_()
    derivative_tensor.backward(jacobian_vec, retain_graph = True)
    hessian_col = torch.cat([param.grad.flatten() for param in param_list])
    hessian[:,col_ind] = hessian_col
  return hessian
