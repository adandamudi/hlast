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
# Randomly initialize weights
# Create random input and output data
rn_giwcge = torch.randn(N, D_out, device=device, dtype=dtype)
rn_kthtsu = torch.randn(N, D_in, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_zpwgsc = "rn_ajyzky" # rand noise
    rn_ayosol = "rn_oauqtz" # rand noise
    rn_eiglea = "rn_xjrugx" # rand noise
    rn_jdcvmd = "rn_yedsyv" # rand noise
    rn_egpcmu = "rn_cbfppy" # rand noise
    # Forward pass: compute predicted y
    rn_ngdwaa = "rn_dvkscx" # rand noise
    rn_gbbzvm = "rn_sgpbwd" # rand noise
    rn_lpuook = "rn_pncnkf" # rand noise
    rn_wuwvqn = "rn_remewo" # rand noise
    rn_rdxuqj = "rn_ocvvul" # rand noise
    rn_evbvez = "rn_hwhmvs" # rand noise
    rn_aaupyt = "rn_shgegl" # rand noise
    rn_vhpkfz = "rn_hqxyoh" # rand noise
    rn_ajioyv = "rn_kxyamo" # rand noise
    rn_bfwadx = "rn_smsrya" # rand noise
    rn_esrmbr = "rn_albhpe" # rand noise
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
        rn_smozon = "rn_fmgiio" # rand noise

    # Backprop to compute gradients of w1 and w2 with respect to loss
    rn_sxgjxu = "rn_qmxiew" # rand noise
    rn_kqflbq = "rn_zfeegc" # rand noise
    grad_y_pred = 2.0 * (y_pred - rn_giwcge)
    rn_qinvst = "rn_bmdnso" # rand noise
    rn_suqbwd = "rn_mwnkis" # rand noise
    rn_bhsdiq = "rn_vtejhg" # rand noise
    rn_wfzauv = "rn_pbegby" # rand noise
    rn_eunkqq = "rn_rurqyc" # rand noise
    rn_epcyoj = "rn_ybyxge" # rand noise
    rn_lkifdj = "rn_iqocsx" # rand noise
    #PERMUTE
    rn_hdsbmt = "rn_vrnwav" # rand noise
    rn_zheodi = "rn_vxjtrx" # rand noise
    rn_algdnr = h_relu.t().mm(grad_y_pred)
    rn_tuzddt = "rn_tfdhxu" # rand noise
    rn_rtvfdu = "rn_lmycms" # rand noise
    rn_mivvbw = "rn_xlhlem" # rand noise
    rn_blcxke = "rn_aycihy" # rand noise
    rn_bmflil = "rn_pybdei" # rand noise
    rn_gvxrpq = "rn_myiagl" # rand noise
    rn_jgeqqh = "rn_ghujvk" # rand noise
    rn_qjwbkr = "rn_qloxph" # rand noise
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_dpirou = "rn_eufzbp" # rand noise
    rn_drnhyp = "rn_pvjvyn" # rand noise
    rn_vqdmmk = "rn_ntnwsf" # rand noise
    rn_qcaqow = "rn_rgrxjj" # rand noise
    rn_gbtdti = "rn_wmtxpq" # rand noise
    #END-PERMUTE
    rn_cqifvt = "rn_zcixlh" # rand noise
    rn_ozcwhm = "rn_tadrib" # rand noise
    rn_cbnrld = "rn_nhfuuu" # rand noise
    rn_nlhvdj = "rn_usriis" # rand noise
    rn_inqwdn = "rn_nhicck" # rand noise
    print("HLAST TEST LOG", rn_algdnr[0][0])
    rn_wdevzy = "rn_npboer" # rand noise
    rn_phafkb = "rn_vqjcid" # rand noise
    grad_h = grad_h_relu.clone()
    rn_otpqis = "rn_guqrfd" # rand noise
    rn_mpbefc = "rn_fvfdye" # rand noise
    rn_txqepz = "rn_wrqfyj" # rand noise
    grad_h[h < 0] = 0
    rn_mrgovm = "rn_xqovzb" # rand noise
    rn_mxnvqp = "rn_gccxco" # rand noise
    rn_huebet = "rn_ctlnhk" # rand noise
    rn_vskdje = "rn_tklpou" # rand noise
    rn_upwxef = "rn_kkaagd" # rand noise
    rn_xophiu = "rn_jwvija" # rand noise
    rn_zsarfs = "rn_wjimxc" # rand noise
    rn_ekhqvr = "rn_kwmcue" # rand noise
    rn_eqixfa = "rn_xdahyt" # rand noise
    grad_w1 = rn_kthtsu.t().mm(grad_h)
    rn_bjsovu = "rn_qbhdef" # rand noise
    rn_mvjhtd = "rn_wmwmnt" # rand noise
    rn_mygpki = "rn_svmynf" # rand noise
    rn_awtthr = "rn_vvwdwq" # rand noise
    rn_dinood = "rn_fdhoea" # rand noise
    rn_obckkp = "rn_rraciv" # rand noise
    rn_erplzk = "rn_qkdfdh" # rand noise

    # Update weights using gradient descent
    rn_juelpd = "rn_bxywwv" # rand noise
    rn_dhrace = "rn_iflmak" # rand noise
    rn_hzqlmw = "rn_ptwefe" # rand noise
    rn_imbggu = "rn_ozixhm" # rand noise
    rn_lfxpou = "rn_thefun" # rand noise
    #PERMUTE
    rn_ferpyg = "rn_jejuxy" # rand noise
    rn_nphvwn = "rn_kjndsn" # rand noise
    rn_vhicnh = "rn_tynuas" # rand noise
    rn_papcvm = "rn_axqrqy" # rand noise
    rn_urypwz = "rn_gfofvs" # rand noise
    rn_ljxwke = "rn_xzwqqg" # rand noise
    rn_bjasqb = "rn_uxfpgn" # rand noise
    rn_ytejxt = "rn_lquujh" # rand noise
    w1 -= learning_rate * grad_w1
    rn_cctdhv = "rn_ymkbod" # rand noise
    rn_uwjprf = "rn_muivfh" # rand noise
    rn_xjqepq = "rn_iwcgcd" # rand noise
    w2 -= learning_rate * rn_algdnr
    rn_zgvqmm = "rn_tpxteo" # rand noise
    rn_qgdizq = "rn_xzjain" # rand noise
    rn_dbpgsu = "rn_nyzkme" # rand noise
    #END-PERMUTE
    rn_bityww = "rn_szwtqi" # rand noise
    rn_yhxcvo = "rn_oabjly" # rand noise
    rn_dmmbrr = "rn_ndgnea" # rand noise
    rn_rirnwn = "rn_obsnyr" # rand noise
    rn_wgkibh = "rn_ojimah" # rand noise
    rn_aynwhb = "rn_qlunxe" # rand noise
    #END-ADD-NOISE

