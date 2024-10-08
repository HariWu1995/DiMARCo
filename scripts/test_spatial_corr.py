import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler, spatial_correlation_sample

device = "cuda"
dtype = torch.float32


def test_lib():
    B = 1    # batch_size
    C = 1    # num_channels
    H = 10
    W = 10

    input1 = torch.randint(1, 4, (B, C, H, W), dtype=dtype, device=device, requires_grad=False)
    input2 = torch.randint_like(input1, 1, 4).requires_grad_(False)

    conv_args = dict(kernel_size=3, stride=2, 
                        padding=0, dilation=2, 
                    patch_size=1, dilation_patch=1)

    # function
    out = spatial_correlation_sample(input1, input2, **conv_args)
    print(out)

    # module
    correlation_sampler = SpatialCorrelationSampler(**conv_args)
    out = correlation_sampler(input1, input2)
    print(out)


def test_arc():

    import arckit
    train_set, eval_set = arckit.load_data()

    task = train_set['05f2a901']
    sample_in, sample_out = task.train[0]

    B, C = 1, 1
    H, W = sample_in.shape

    sample_in  = torch.tensor(sample_in ).to(device).to(dtype).unsqueeze(dim=0).unsqueeze(dim=0)
    sample_out = torch.tensor(sample_out).to(device).to(dtype).unsqueeze(dim=0).unsqueeze(dim=0)
    print(sample_in, '\n')
    print(sample_out, '\n\n')

    conv_args = dict(kernel_size=1, stride=1, 
                        padding=0, dilation=1, 
                    patch_size=1, dilation_patch=1)

    corr = spatial_correlation_sample(sample_in, sample_out, **conv_args)
    
    _B, pH, pW, _H, _W = corr.size()
    corr = corr.view(B, pH * pW, H, W) / sample_in.size(1)
    corr = F.leaky_relu_(corr, 0.1)
    print(corr)
        

if __name__ == '__main__':
    
    # test_lib()
    test_arc()

