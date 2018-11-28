import torch
from torch.autograd import Variable


# pytorch中的tensor
# x = torch.Tensor(5, 3)
# print(x, x.shape, x.size())
# x = torch.rand(5, 3)
# print(x, x.shape, x.size())
# x_np = x.numpy()
# print(x_np, type(x_np))

# pytorch中的Variable
# x = Variable(torch.ones(2, 2), requires_grad=True)
# # print(x)
# y = x + 2
# # print(y)
# # print(x.grad_fn)    # grad_fn指向创建该变量的function，由于x是由用户创建的,所以x的grad_fn为None
# # print(y.grad_fn)
# # pytorch中的梯度Gradients
# z = y * y * 3
# out = z.mean()
# print(z, out)
# out.backward()  # 执行反向传播
# print(x.grad)
