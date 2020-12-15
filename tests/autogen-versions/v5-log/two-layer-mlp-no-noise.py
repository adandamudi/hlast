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
# Create random input and output data
y_vals = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# Randomly initialize weights
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
x_vals = torch.randn(N, D_in, device=device, dtype=dtype)
#END-PERMUTE

learning_rate = 1e-6
for t in range(500):
    #ADD-NOISE
    # Forward pass: compute predicted y
    h = x_vals.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y_vals).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y_vals)
    #PERMUTE
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    #END-PERMUTE
    print("HLAST TEST LOG", grad_w2[0][0])
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x_vals.t().mm(grad_h)

    # Update weights using gradient descent
    #PERMUTE
    w2 -= learning_rate * grad_w2
    w1 -= learning_rate * grad_w1
    #END-PERMUTE
    #END-ADD-NOISE

