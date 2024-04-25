import torch


if __name__ == '__main__':
    torch.ops.load_library("build/libmy_add.so")
    a = torch.rand([1,3]).cuda()
    b = torch.rand([1,3]).cuda()
    c = a + b
    print("c: ", c)
    d = torch.empty(3).to(device="cuda:0")
    torch.ops.my_add.torch_launch_my_add(d, a, b, 3)
    d = d.to("cpu")
    print("d after: ", d)



