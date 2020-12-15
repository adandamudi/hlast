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
device = torch.device(dev)
dtype = torch.float
#END-PERMUTE

#PERMUTE
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
rn_kthtsu = torch.randn(N, D_in, device=device, dtype=dtype)
# Randomly initialize weights
# Create random input and output data
rn_giwcge = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_eiglea = "rn_xjrugx" # rand noise
    # Forward pass: compute predicted y
    rn_wuwvqn = "rn_remewo" # rand noise
    rn_rdxuqj = "rn_ocvvul" # rand noise
    rn_vhpkfz = "rn_hqxyoh" # rand noise
    h = rn_kthtsu.mm(w1)
    h_relu = h.clamp(min=0)
    rn_aimajx = "rn_xawzlx" # rand noise
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - rn_giwcge).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - rn_giwcge)
    rn_wfzauv = "rn_pbegby" # rand noise
    #PERMUTE
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_algdnr = h_relu.t().mm(grad_y_pred)
    rn_qjwbkr = "rn_qloxph" # rand noise
    rn_blcxke = "rn_aycihy" # rand noise
    rn_gvxrpq = "rn_myiagl" # rand noise
    rn_vqdmmk = "rn_ntnwsf" # rand noise
    #END-PERMUTE
    rn_cqifvt = "rn_zcixlh" # rand noise
    rn_cbnrld = "rn_nhfuuu" # rand noise
    grad_h = grad_h_relu.clone()
    rn_otpqis = "rn_guqrfd" # rand noise
    grad_h[h < 0] = 0
    rn_huebet = "rn_ctlnhk" # rand noise
    rn_zsarfs = "rn_wjimxc" # rand noise
    grad_w1 = rn_kthtsu.t().mm(grad_h)
    rn_mvjhtd = "rn_wmwmnt" # rand noise
    rn_awtthr = "rn_vvwdwq" # rand noise

    # Update weights using gradient descent
    rn_hzqlmw = "rn_ptwefe" # rand noise
    #PERMUTE
    rn_uwjprf = "rn_muivfh" # rand noise
    rn_papcvm = "rn_axqrqy" # rand noise
    w1 -= learning_rate * grad_w1
    rn_qgdizq = "rn_xzjain" # rand noise
    w2 -= learning_rate * rn_algdnr
    #END-PERMUTE
    rn_dmmbrr = "rn_ndgnea" # rand noise
    #END-ADD-NOISE

