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
rn_kugogp = torch.randn(N, D_in, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
rn_poiekh = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# Create random input and output data
# Randomly initialize weights
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_aqcorl = "rn_orzxfc" # rand noise
    # Forward pass: compute predicted y
    h = rn_kugogp.mm(w1)
    rn_dzuivw = "rn_zhqrso" # rand noise
    h_relu = h.clamp(min=0)
    rn_ykcawd = "rn_hvjviw" # rand noise
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - rn_poiekh).pow(2).sum().item()
    rn_jtzmec = "rn_fwtztf" # rand noise
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    rn_ebaauj = "rn_wfzzew" # rand noise
    grad_y_pred = 2.0 * (y_pred - rn_poiekh)
    #PERMUTE
    rn_dbvich = h_relu.t().mm(grad_y_pred)
    rn_svjabe = "rn_obzslq" # rand noise
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_vpkdnr = "rn_wggejs" # rand noise
    #END-PERMUTE
    print("HLAST TEST LOG", rn_dbvich[0][0])
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = rn_kugogp.t().mm(grad_h)
    rn_xfkxqa = "rn_jnhlxi" # rand noise

    # Update weights using gradient descent
    rn_ylklji = "rn_sxxotg" # rand noise
    #PERMUTE
    rn_vvouwe = "rn_apgnpk" # rand noise
    w2 -= learning_rate * rn_dbvich
    w1 -= learning_rate * grad_w1
    rn_ecwavy = "rn_hedmdj" # rand noise
    #END-PERMUTE
    #END-ADD-NOISE

