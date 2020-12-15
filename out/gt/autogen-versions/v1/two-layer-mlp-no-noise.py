import torch
(N, D_in, H, D_out) = (64, 1000, 100, 10)
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
dtype = torch.float
device = torch.device(dev)
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
learning_rate = 1e-06
for t in range(500):
    h = x_vals.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y_vals).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)
    grad_y_pred = 2.0 * (y_pred - y_vals)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    print('HLAST TEST LOG', grad_w2[0][0])
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x_vals.t().mm(grad_h)
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
