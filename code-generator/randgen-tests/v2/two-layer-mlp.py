#RENAME x_vals y_vals grad_w2

import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# run on gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

#PERMUTE
dtype = torch.float
device = torch.device(dev)
#END-PERMUTE

#PERMUTE
rn_kthtsu = torch.randn(N, D_in, device=device, dtype=dtype)
# Create random input and output data
# Randomly initialize weights
rn_giwcge = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_eiglea = "rn_xjrugx" # rand noise
    # Forward pass: compute predicted y
    rn_rdxuqj = "rn_ocvvul" # rand noise
    h = rn_kthtsu.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - rn_giwcge).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - rn_giwcge)
    #PERMUTE
    rn_qjwbkr = "rn_qloxph" # rand noise
    rn_algdnr = h_relu.t().mm(grad_y_pred)
    rn_gvxrpq = "rn_myiagl" # rand noise
    grad_h_relu = grad_y_pred.mm(w2.t())
    #END-PERMUTE
    rn_cbnrld = "rn_nhfuuu" # rand noise
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    rn_zsarfs = "rn_wjimxc" # rand noise
    grad_w1 = rn_kthtsu.t().mm(grad_h)
    rn_mvjhtd = "rn_wmwmnt" # rand noise

    # Update weights using gradient descent
    #PERMUTE
    w1 -= learning_rate * grad_w1
    rn_papcvm = "rn_axqrqy" # rand noise
    w2 -= learning_rate * rn_algdnr
    rn_uwjprf = "rn_muivfh" # rand noise
    #END-PERMUTE
    #END-ADD-NOISE

