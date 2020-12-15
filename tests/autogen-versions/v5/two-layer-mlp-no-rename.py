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
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
# Randomly initialize weights
# Create random input and output data
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    rn_orydhu = "rn_jrpatc" # rand noise
    rn_lhntvh = "rn_igkhxg" # rand noise
    # Forward pass: compute predicted y
    rn_tgglso = "rn_tcawjf" # rand noise
    rn_funtgt = "rn_quurfb" # rand noise
    rn_zmpatb = "rn_vwifuh" # rand noise
    rn_fdvuvh = "rn_xvwkch" # rand noise
    rn_jxrkby = "rn_ryqzbj" # rand noise
    h = x_vals.mm(w1)
    rn_pkpoco = "rn_hdcnnu" # rand noise
    rn_xsslou = "rn_hvoymy" # rand noise
    h_relu = h.clamp(min=0)
    rn_icynif = "rn_peskot" # rand noise
    rn_ixtzjs = "rn_gtwoqj" # rand noise
    rn_shcaia = "rn_bsnukw" # rand noise
    rn_wjmctk = "rn_fcaxnf" # rand noise
    rn_nuihwk = "rn_mybwka" # rand noise
    rn_uxrmdq = "rn_toildx" # rand noise
    y_pred = h_relu.mm(w2)
    rn_rasydh = "rn_luhvpj" # rand noise
    rn_mekrup = "rn_maclcv" # rand noise
    rn_bhglnw = "rn_bkjbpa" # rand noise

    # Compute and print loss
    rn_mwxple = "rn_oqaslq" # rand noise
    rn_crwsdj = "rn_gzcqkg" # rand noise
    loss = (y_pred - y_vals).pow(2).sum().item()
    rn_wzlllv = "rn_ikqbcu" # rand noise
    rn_irweos = "rn_ridkbi" # rand noise
    rn_ffgdex = "rn_byumdg" # rand noise
    rn_dyrybl = "rn_hmfnlt" # rand noise
    rn_sykrhu = "rn_ylsbpe" # rand noise
    rn_bqtcjy = "rn_bxrbun" # rand noise
    rn_bnsxwy = "rn_cjsyfy" # rand noise
    if t % 100 == 99:
        print(t, loss)
        rn_vykdsr = "rn_rylsqf" # rand noise
        rn_wsfoob = "rn_yxwtyx" # rand noise
        rn_yduncy = "rn_ecrpmj" # rand noise
        rn_qoigyt = "rn_ugskvq" # rand noise
        rn_gybwhw = "rn_kbvlaw" # rand noise
        rn_exmjuv = "rn_njlbkp" # rand noise

    # Backprop to compute gradients of w1 and w2 with respect to loss
    rn_uyzsjp = "rn_fuhmrt" # rand noise
    rn_vaqahs = "rn_txnhyg" # rand noise
    rn_yykhxc = "rn_lslpck" # rand noise
    rn_ynxwaj = "rn_ffublg" # rand noise
    rn_sowubu = "rn_uxmlse" # rand noise
    rn_uaybnu = "rn_slbulh" # rand noise
    grad_y_pred = 2.0 * (y_pred - y_vals)
    rn_iwilsv = "rn_lljyiy" # rand noise
    rn_zxdaqh = "rn_duempu" # rand noise
    rn_cuvwjt = "rn_jitxgz" # rand noise
    rn_tohdri = "rn_ithvim" # rand noise
    rn_nuorxp = "rn_etkafv" # rand noise
    rn_buguxf = "rn_aygkuy" # rand noise
    rn_kmdbfw = "rn_ykzifa" # rand noise
    #PERMUTE
    rn_zussqq = "rn_mqqher" # rand noise
    rn_uphdxg = "rn_mnxoxx" # rand noise
    grad_w2 = h_relu.t().mm(grad_y_pred)
    rn_ryzqhw = "rn_kfbwyo" # rand noise
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_gdusut = "rn_flrdax" # rand noise
    rn_mibdvs = "rn_gvqzxw" # rand noise
    rn_fgtokm = "rn_dyovwu" # rand noise
    rn_pjspml = "rn_jootuc" # rand noise
    rn_ykjnty = "rn_tvlkvv" # rand noise
    rn_cpmmih = "rn_ruuoel" # rand noise
    rn_enlbix = "rn_fwtbeb" # rand noise
    rn_bjmvjd = "rn_gwszqb" # rand noise
    rn_smprae = "rn_sbakru" # rand noise
    rn_hdofrx = "rn_pzrlwy" # rand noise
    rn_wbgzjt = "rn_tgsawt" # rand noise
    rn_aoxkxg = "rn_raxlgz" # rand noise
    rn_svsgsu = "rn_vollhl" # rand noise
    rn_bpodtj = "rn_sihcnh" # rand noise
    #END-PERMUTE
    rn_bpbjht = "rn_rdjooc" # rand noise
    rn_njuhqc = "rn_ztdlcv" # rand noise
    rn_ncjgtl = "rn_nkpbzt" # rand noise
    rn_tleejh = "rn_rqxpfz" # rand noise
    rn_wbkxej = "rn_nemjaq" # rand noise
    rn_osrzor = "rn_pssgai" # rand noise
    rn_zdifru = "rn_nhkpwd" # rand noise
    rn_svlxex = "rn_btqeej" # rand noise
    grad_h = grad_h_relu.clone()
    rn_rbnjfw = "rn_mxwzlz" # rand noise
    rn_ekkhbt = "rn_sjnevt" # rand noise
    rn_mclxtv = "rn_dwweux" # rand noise
    rn_rwxtwi = "rn_nuzcwh" # rand noise
    rn_qvmfdg = "rn_gqzgcs" # rand noise
    grad_h[h < 0] = 0
    rn_zmrltj = "rn_ulqora" # rand noise
    rn_jlonqp = "rn_tgcmno" # rand noise
    rn_rheyzt = "rn_cyqekb" # rand noise
    grad_w1 = x_vals.t().mm(grad_h)

    # Update weights using gradient descent
    rn_oiowxv = "rn_xelukv" # rand noise
    rn_vdecrg = "rn_ibksfi" # rand noise
    #PERMUTE
    rn_dxryqo = "rn_icfxon" # rand noise
    rn_ictuih = "rn_ucktqv" # rand noise
    w2 -= learning_rate * grad_w2
    rn_tirdpe = "rn_gqjswt" # rand noise
    rn_kkenhc = "rn_rshvle" # rand noise
    rn_vueebe = "rn_adcuwm" # rand noise
    rn_kbkqjh = "rn_seubnv" # rand noise
    rn_uodion = "rn_ncvukw" # rand noise
    w1 -= learning_rate * grad_w1
    rn_grkxwy = "rn_iygipu" # rand noise
    rn_dbjwzc = "rn_pjqryy" # rand noise
    #END-PERMUTE
    rn_kjtdph = "rn_ahyetx" # rand noise
    rn_kwnusl = "rn_joedmq" # rand noise
    rn_mrzeuc = "rn_qrdklh" # rand noise
    rn_iqtiur = "rn_ermlfh" # rand noise
    #END-ADD-NOISE

