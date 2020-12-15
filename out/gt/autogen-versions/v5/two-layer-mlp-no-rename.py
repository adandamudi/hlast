import torch
(N, D_in, H, D_out) = (64, 1000, 100, 10)
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)
dtype = torch.float
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
learning_rate = 1e-06
for t in range(500):
    rn_orydhu = 'rn_jrpatc'
    rn_lhntvh = 'rn_igkhxg'
    rn_tgglso = 'rn_tcawjf'
    rn_funtgt = 'rn_quurfb'
    rn_zmpatb = 'rn_vwifuh'
    rn_fdvuvh = 'rn_xvwkch'
    rn_jxrkby = 'rn_ryqzbj'
    h = x_vals.mm(w1)
    rn_pkpoco = 'rn_hdcnnu'
    rn_xsslou = 'rn_hvoymy'
    h_relu = h.clamp(min=0)
    rn_icynif = 'rn_peskot'
    rn_ixtzjs = 'rn_gtwoqj'
    rn_shcaia = 'rn_bsnukw'
    rn_wjmctk = 'rn_fcaxnf'
    rn_nuihwk = 'rn_mybwka'
    rn_uxrmdq = 'rn_toildx'
    y_pred = h_relu.mm(w2)
    rn_rasydh = 'rn_luhvpj'
    rn_mekrup = 'rn_maclcv'
    rn_bhglnw = 'rn_bkjbpa'
    rn_mwxple = 'rn_oqaslq'
    rn_crwsdj = 'rn_gzcqkg'
    loss = (y_pred - y_vals).pow(2).sum().item()
    rn_wzlllv = 'rn_ikqbcu'
    rn_irweos = 'rn_ridkbi'
    rn_ffgdex = 'rn_byumdg'
    rn_dyrybl = 'rn_hmfnlt'
    rn_sykrhu = 'rn_ylsbpe'
    rn_bqtcjy = 'rn_bxrbun'
    rn_bnsxwy = 'rn_cjsyfy'
    if t % 100 == 99:
        print(t, loss)
        rn_vykdsr = 'rn_rylsqf'
        rn_wsfoob = 'rn_yxwtyx'
        rn_yduncy = 'rn_ecrpmj'
        rn_qoigyt = 'rn_ugskvq'
        rn_gybwhw = 'rn_kbvlaw'
        rn_exmjuv = 'rn_njlbkp'
    rn_uyzsjp = 'rn_fuhmrt'
    rn_vaqahs = 'rn_txnhyg'
    rn_yykhxc = 'rn_lslpck'
    rn_ynxwaj = 'rn_ffublg'
    rn_sowubu = 'rn_uxmlse'
    rn_uaybnu = 'rn_slbulh'
    grad_y_pred = 2.0 * (y_pred - y_vals)
    rn_iwilsv = 'rn_lljyiy'
    rn_zxdaqh = 'rn_duempu'
    rn_cuvwjt = 'rn_jitxgz'
    rn_tohdri = 'rn_ithvim'
    rn_nuorxp = 'rn_etkafv'
    rn_buguxf = 'rn_aygkuy'
    rn_kmdbfw = 'rn_ykzifa'
    rn_zussqq = 'rn_mqqher'
    rn_uphdxg = 'rn_mnxoxx'
    grad_w2 = h_relu.t().mm(grad_y_pred)
    rn_ryzqhw = 'rn_kfbwyo'
    grad_h_relu = grad_y_pred.mm(w2.t())
    rn_gdusut = 'rn_flrdax'
    rn_mibdvs = 'rn_gvqzxw'
    rn_fgtokm = 'rn_dyovwu'
    rn_pjspml = 'rn_jootuc'
    rn_ykjnty = 'rn_tvlkvv'
    rn_cpmmih = 'rn_ruuoel'
    rn_enlbix = 'rn_fwtbeb'
    rn_bjmvjd = 'rn_gwszqb'
    rn_smprae = 'rn_sbakru'
    rn_hdofrx = 'rn_pzrlwy'
    rn_wbgzjt = 'rn_tgsawt'
    rn_aoxkxg = 'rn_raxlgz'
    rn_svsgsu = 'rn_vollhl'
    rn_bpodtj = 'rn_sihcnh'
    rn_bpbjht = 'rn_rdjooc'
    rn_njuhqc = 'rn_ztdlcv'
    rn_ncjgtl = 'rn_nkpbzt'
    rn_tleejh = 'rn_rqxpfz'
    rn_wbkxej = 'rn_nemjaq'
    print('HLAST TEST LOG', grad_w2[0][0])
    rn_osrzor = 'rn_pssgai'
    rn_zdifru = 'rn_nhkpwd'
    rn_svlxex = 'rn_btqeej'
    grad_h = grad_h_relu.clone()
    rn_rbnjfw = 'rn_mxwzlz'
    rn_ekkhbt = 'rn_sjnevt'
    rn_mclxtv = 'rn_dwweux'
    rn_rwxtwi = 'rn_nuzcwh'
    rn_qvmfdg = 'rn_gqzgcs'
    grad_h[h < 0] = 0
    rn_zmrltj = 'rn_ulqora'
    rn_jlonqp = 'rn_tgcmno'
    rn_rheyzt = 'rn_cyqekb'
    grad_w1 = x_vals.t().mm(grad_h)
    rn_oiowxv = 'rn_xelukv'
    rn_vdecrg = 'rn_ibksfi'
    rn_dxryqo = 'rn_icfxon'
    rn_ictuih = 'rn_ucktqv'
    w2 -= learning_rate * grad_w2
    rn_tirdpe = 'rn_gqjswt'
    rn_kkenhc = 'rn_rshvle'
    rn_vueebe = 'rn_adcuwm'
    rn_kbkqjh = 'rn_seubnv'
    rn_uodion = 'rn_ncvukw'
    w1 -= learning_rate * grad_w1
    rn_grkxwy = 'rn_iygipu'
    rn_dbjwzc = 'rn_pjqryy'
    rn_kjtdph = 'rn_ahyetx'
    rn_kwnusl = 'rn_joedmq'
    rn_mrzeuc = 'rn_qrdklh'
    rn_iqtiur = 'rn_ermlfh'
