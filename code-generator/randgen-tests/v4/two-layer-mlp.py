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
rn_kthtsu = torch.randn(N, D_in, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# Create random input and output data
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
# Randomly initialize weights
rn_giwcge = torch.randn(N, D_out, device=device, dtype=dtype)
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_zpwgsc = "rn_ajyzky" # rand noise
    rn_eiglea = "rn_xjrugx" # rand noise
    rn_jdcvmd = "rn_yedsyv" # rand noise
    # Forward pass: compute predicted y
    rn_gbbzvm = "rn_sgpbwd" # rand noise
    rn_wuwvqn = "rn_remewo" # rand noise
    rn_rdxuqj = "rn_ocvvul" # rand noise
    rn_aaupyt = "rn_shgegl" # rand noise
    rn_vhpkfz = "rn_hqxyoh" # rand noise
    rn_bfwadx = "rn_smsrya" # rand noise
    h = rn_kthtsu.mm(w1)
    rn_fescoz = "rn_bcfcgy" # rand noise
    h_relu = h.clamp(min=0)
    rn_aimajx = "rn_xawzlx" # rand noise
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    rn_zjytfz = "rn_idxede" # rand noise
    loss = (y_pred - rn_giwcge).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    rn_sxgjxu = "rn_qmxiew" # rand noise
    grad_y_pred = 2.0 * (y_pred - rn_giwcge)
    rn_suqbwd = "rn_mwnkis" # rand noise
    rn_wfzauv = "rn_pbegby" # rand noise
    rn_epcyoj = "rn_ybyxge" # rand noise
    #PERMUTE
    rn_rtvfdu = "rn_lmycms" # rand noise
    rn_gvxrpq = "rn_myiagl" # rand noise
    rn_hdsbmt = "rn_vrnwav" # rand noise
    rn_qjwbkr = "rn_qloxph" # rand noise
    rn_algdnr = h_relu.t().mm(grad_y_pred)
    rn_vqdmmk = "rn_ntnwsf" # rand noise
    rn_drnhyp = "rn_pvjvyn" # rand noise
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_blcxke = "rn_aycihy" # rand noise
    rn_qcaqow = "rn_rgrxjj" # rand noise
    #END-PERMUTE
    rn_cqifvt = "rn_zcixlh" # rand noise
    rn_cbnrld = "rn_nhfuuu" # rand noise
    rn_nlhvdj = "rn_usriis" # rand noise
    rn_wdevzy = "rn_npboer" # rand noise
    grad_h = grad_h_relu.clone()
    rn_otpqis = "rn_guqrfd" # rand noise
    rn_txqepz = "rn_wrqfyj" # rand noise
    grad_h[h < 0] = 0
    rn_mxnvqp = "rn_gccxco" # rand noise
    rn_huebet = "rn_ctlnhk" # rand noise
    rn_upwxef = "rn_kkaagd" # rand noise
    rn_zsarfs = "rn_wjimxc" # rand noise
    rn_eqixfa = "rn_xdahyt" # rand noise
    grad_w1 = rn_kthtsu.t().mm(grad_h)
    rn_bjsovu = "rn_qbhdef" # rand noise
    rn_mvjhtd = "rn_wmwmnt" # rand noise
    rn_awtthr = "rn_vvwdwq" # rand noise
    rn_obckkp = "rn_rraciv" # rand noise

    # Update weights using gradient descent
    rn_dhrace = "rn_iflmak" # rand noise
    rn_hzqlmw = "rn_ptwefe" # rand noise
    rn_imbggu = "rn_ozixhm" # rand noise
    #PERMUTE
    rn_ljxwke = "rn_xzwqqg" # rand noise
    rn_uwjprf = "rn_muivfh" # rand noise
    rn_ytejxt = "rn_lquujh" # rand noise
    rn_qgdizq = "rn_xzjain" # rand noise
    rn_nphvwn = "rn_kjndsn" # rand noise
    w1 -= learning_rate * grad_w1
    rn_papcvm = "rn_axqrqy" # rand noise
    w2 -= learning_rate * rn_algdnr
    rn_cctdhv = "rn_ymkbod" # rand noise
    #END-PERMUTE
    rn_bityww = "rn_szwtqi" # rand noise
    rn_dmmbrr = "rn_ndgnea" # rand noise
    rn_wgkibh = "rn_ojimah" # rand noise
    #END-ADD-NOISE

