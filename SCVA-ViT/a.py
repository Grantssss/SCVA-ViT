import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count()) #查看可行的cuda数目

x = torch.rand(5,3)
print(x)
