def Hessian_vector_prod(loss, my_net):
  
  import copy
  my_net.zero_grad()
  para_store = [param for param in my_net.parameters()]
  gradf = torch.autograd.grad(loss, para_store,create_graph=True)
  store_grad = gradf
  for idx, param in enumerate(store_grad):
    if idx == 0:
      store_grad_n = torch.flatten(param)
    else:
      store_grad_n = torch.cat((store_grad_n, torch.flatten(param)))
  vec = copy.copy(store_grad_n).detach()
  g = torch.dot(store_grad_n, vec)
  g.backward()
  
  final_product = []
  for idx, param in enumerate(my_net.parameters()):
    if idx == 0:
      final_product = param.grad.flatten()
    else:
      final_product = torch.cat((final_product, torch.flatten(param.grad)))
  return final_product
