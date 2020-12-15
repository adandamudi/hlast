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
rn_ijeujj = torch.randn(N, D_in, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# Create random input and output data
rn_vvhfnv = torch.randn(N, D_out, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
# Randomly initialize weights
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_cpxzvs = "rn_kubhqs" # rand noise
    # Forward pass: compute predicted y
    h = rn_ijeujj.mm(w1)
    rn_vfuhrj = "rn_fvsrhe" # rand noise
    h_relu = h.clamp(min=0)
    rn_qebeiz = "rn_ewvnll" # rand noise
    y_pred = h_relu.mm(w2)
    rn_awxlpm = "rn_yaueuc" # rand noise

    # Compute and print loss
    rn_vuylkg = "rn_hbouef" # rand noise
    rn_fiuyol = "rn_qrsegt" # rand noise
    rn_umyavq = "rn_mhmxfg" # rand noise
    loss = (y_pred - rn_vvhfnv).pow(2).sum().item()
    rn_dqcuxu = "rn_narotx" # rand noise
    if t % 100 == 99:
        print(t, loss)
        rn_gqdgyb = "rn_lnoaqp" # rand noise
        rn_cevxoa = "rn_baxsyu" # rand noise

    # Backprop to compute gradients of w1 and w2 with respect to loss
    rn_ejdhwn = "rn_okzajw" # rand noise
    rn_sxvbsk = "rn_xrzzis" # rand noise
    grad_y_pred = 2.0 * (y_pred - rn_vvhfnv)
    rn_prohuv = "rn_xnbyqz" # rand noise
    #PERMUTE
    rn_mhakor = "rn_weernm" # rand noise
    rn_yzylis = "rn_ofzueu" # rand noise
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_sqbdjl = "rn_srhwhq" # rand noise
    rn_iymnnh = "rn_pmavkd" # rand noise
    rn_wrwmra = h_relu.t().mm(grad_y_pred)
    rn_iohznm = "rn_azuzqm" # rand noise
    #END-PERMUTE
    rn_nhsynb = "rn_keylvz" # rand noise
    grad_h = grad_h_relu.clone()
    rn_yawfnl = "rn_vsdipe" # rand noise
    rn_auzreg = "rn_kxighu" # rand noise
    grad_h[h < 0] = 0
    rn_oumuyu = "rn_timopq" # rand noise
    rn_wpaipx = "rn_hvymqu" # rand noise
    grad_w1 = rn_ijeujj.t().mm(grad_h)
    rn_pdrotu = "rn_xkohqi" # rand noise
    rn_tbfklq = "rn_vrtjhu" # rand noise
    rn_sqeino = "rn_czigxo" # rand noise

    # Update weights using gradient descent
    rn_hcwntk = "rn_wijvlo" # rand noise
    rn_jdbgsg = "rn_dteabr" # rand noise
    #PERMUTE
    rn_rbxuhr = "rn_htnvup" # rand noise
    rn_pmyuoj = "rn_wedlam" # rand noise
    w2 -= learning_rate * rn_wrwmra
    w1 -= learning_rate * grad_w1
    #END-PERMUTE
    rn_qvdlnn = "rn_wzxgxd" # rand noise
    #END-ADD-NOISE

