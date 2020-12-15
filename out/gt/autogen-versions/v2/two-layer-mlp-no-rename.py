import torch
(N, D_in, H, D_out) = (64, 1000, 100, 10)
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)
dtype = torch.float
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
learning_rate = 1e-06
for t in range(500):
    rn_zmpatb = 'rn_vwifuh'
    h = x_vals.mm(w1)
    rn_xsslou = 'rn_hvoymy'
    h_relu = h.clamp(min=0)
    rn_wjmctk = 'rn_fcaxnf'
    y_pred = h_relu.mm(w2)
    rn_rasydh = 'rn_luhvpj'
    rn_crwsdj = 'rn_gzcqkg'
    loss = (y_pred - y_vals).pow(2).sum().item()
    rn_dyrybl = 'rn_hmfnlt'
    if t % 100 == 99:
        print(t, loss)
        rn_wsfoob = 'rn_yxwtyx'
    rn_uyzsjp = 'rn_fuhmrt'
    grad_y_pred = 2.0 * (y_pred - y_vals)
    rn_buguxf = 'rn_aygkuy'
    rn_enlbix = 'rn_fwtbeb'
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_w2 = h_relu.t().mm(grad_y_pred)
    rn_zussqq = 'rn_mqqher'
    rn_ncjgtl = 'rn_nkpbzt'
    print('HLAST TEST LOG', grad_w2[0][0])
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x_vals.t().mm(grad_h)
    rn_oiowxv = 'rn_xelukv'
    w2 -= learning_rate * grad_w2
    w1 -= learning_rate * grad_w1
    rn_grkxwy = 'rn_iygipu'
    rn_kwnusl = 'rn_joedmq'
