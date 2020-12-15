import torch
(N, D_in, H, D_out) = (64, 1000, 100, 10)
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
dtype = torch.float
device = torch.device(dev)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
learning_rate = 1e-06
for t in range(500):
    rn_lhntvh = 'rn_igkhxg'
    rn_funtgt = 'rn_quurfb'
    rn_zmpatb = 'rn_vwifuh'
    rn_jxrkby = 'rn_ryqzbj'
    h = x_vals.mm(w1)
    rn_xsslou = 'rn_hvoymy'
    h_relu = h.clamp(min=0)
    rn_ixtzjs = 'rn_gtwoqj'
    rn_wjmctk = 'rn_fcaxnf'
    rn_uxrmdq = 'rn_toildx'
    y_pred = h_relu.mm(w2)
    rn_rasydh = 'rn_luhvpj'
    rn_mekrup = 'rn_maclcv'
    rn_mwxple = 'rn_oqaslq'
    rn_crwsdj = 'rn_gzcqkg'
    loss = (y_pred - y_vals).pow(2).sum().item()
    rn_wzlllv = 'rn_ikqbcu'
    rn_ffgdex = 'rn_byumdg'
    rn_dyrybl = 'rn_hmfnlt'
    rn_bqtcjy = 'rn_bxrbun'
    if t % 100 == 99:
        print(t, loss)
        rn_wsfoob = 'rn_yxwtyx'
        rn_yduncy = 'rn_ecrpmj'
        rn_gybwhw = 'rn_kbvlaw'
    rn_uyzsjp = 'rn_fuhmrt'
    rn_yykhxc = 'rn_lslpck'
    rn_sowubu = 'rn_uxmlse'
    grad_y_pred = 2.0 * (y_pred - y_vals)
    rn_zxdaqh = 'rn_duempu'
    rn_tohdri = 'rn_ithvim'
    rn_buguxf = 'rn_aygkuy'
    rn_pjspml = 'rn_jootuc'
    rn_enlbix = 'rn_fwtbeb'
    rn_svsgsu = 'rn_vollhl'
    rn_gdusut = 'rn_flrdax'
    rn_wbgzjt = 'rn_tgsawt'
    rn_bpodtj = 'rn_sihcnh'
    rn_smprae = 'rn_sbakru'
    rn_zussqq = 'rn_mqqher'
    rn_cpmmih = 'rn_ruuoel'
    grad_w2 = h_relu.t().mm(grad_y_pred)
    rn_fgtokm = 'rn_dyovwu'
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_njuhqc = 'rn_ztdlcv'
    rn_ncjgtl = 'rn_nkpbzt'
    rn_wbkxej = 'rn_nemjaq'
    print('HLAST TEST LOG', grad_w2[0][0])
    rn_zdifru = 'rn_nhkpwd'
    grad_h = grad_h_relu.clone()
    rn_ekkhbt = 'rn_sjnevt'
    rn_rwxtwi = 'rn_nuzcwh'
    rn_qvmfdg = 'rn_gqzgcs'
    grad_h[h < 0] = 0
    rn_zmrltj = 'rn_ulqora'
    rn_rheyzt = 'rn_cyqekb'
    grad_w1 = x_vals.t().mm(grad_h)
    rn_oiowxv = 'rn_xelukv'
    rn_vdecrg = 'rn_ibksfi'
    rn_tirdpe = 'rn_gqjswt'
    rn_grkxwy = 'rn_iygipu'
    rn_ictuih = 'rn_ucktqv'
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    rn_vueebe = 'rn_adcuwm'
    rn_kbkqjh = 'rn_seubnv'
    rn_dxryqo = 'rn_icfxon'
    rn_kjtdph = 'rn_ahyetx'
    rn_kwnusl = 'rn_joedmq'
    rn_iqtiur = 'rn_ermlfh'
