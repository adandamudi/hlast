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
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# Create random input and output data
# Randomly initialize weights
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_lhntvh = "rn_igkhxg" # rand noise
    # Forward pass: compute predicted y
    rn_funtgt = "rn_quurfb" # rand noise
    rn_zmpatb = "rn_vwifuh" # rand noise
    rn_jxrkby = "rn_ryqzbj" # rand noise
    h = x_vals.mm(w1)
    rn_xsslou = "rn_hvoymy" # rand noise
    h_relu = h.clamp(min=0)
    rn_wjmctk = "rn_fcaxnf" # rand noise
    y_pred = h_relu.mm(w2)
    rn_rasydh = "rn_luhvpj" # rand noise

    # Compute and print loss
    rn_mwxple = "rn_oqaslq" # rand noise
    rn_crwsdj = "rn_gzcqkg" # rand noise
    loss = (y_pred - y_vals).pow(2).sum().item()
    rn_ffgdex = "rn_byumdg" # rand noise
    rn_dyrybl = "rn_hmfnlt" # rand noise
    if t % 100 == 99:
        print(t, loss)
        rn_wsfoob = "rn_yxwtyx" # rand noise
        rn_yduncy = "rn_ecrpmj" # rand noise

    # Backprop to compute gradients of w1 and w2 with respect to loss
    rn_uyzsjp = "rn_fuhmrt" # rand noise
    rn_sowubu = "rn_uxmlse" # rand noise
    grad_y_pred = 2.0 * (y_pred - y_vals)
    rn_zxdaqh = "rn_duempu" # rand noise
    rn_buguxf = "rn_aygkuy" # rand noise
    #PERMUTE
    rn_cpmmih = "rn_ruuoel" # rand noise
    rn_zussqq = "rn_mqqher" # rand noise
    rn_bpodtj = "rn_sihcnh" # rand noise
    rn_enlbix = "rn_fwtbeb" # rand noise
    rn_gdusut = "rn_flrdax" # rand noise
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    #END-PERMUTE
    rn_ncjgtl = "rn_nkpbzt" # rand noise
    rn_wbkxej = "rn_nemjaq" # rand noise
    print("HLAST TEST LOG", grad_w2[0][0])
    grad_h = grad_h_relu.clone()
    rn_rwxtwi = "rn_nuzcwh" # rand noise
    grad_h[h < 0] = 0
    rn_zmrltj = "rn_ulqora" # rand noise
    grad_w1 = x_vals.t().mm(grad_h)

    # Update weights using gradient descent
    rn_oiowxv = "rn_xelukv" # rand noise
    #PERMUTE
    rn_kbkqjh = "rn_seubnv" # rand noise
    w2 -= learning_rate * grad_w2
    rn_grkxwy = "rn_iygipu" # rand noise
    w1 -= learning_rate * grad_w1
    #END-PERMUTE
    rn_kwnusl = "rn_joedmq" # rand noise
    #END-ADD-NOISE

