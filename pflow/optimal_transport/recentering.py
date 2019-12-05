import geomloss as gl
import torch


def recenter_from_proposal(x, w_y, y, lr=0.5, n_iter=50, **kwargs):
    uniform = torch.ones_like(w_y)
    uniform /= uniform.sum()
    x_new = x.clone().requires_grad_(True)
    sample_loss = gl.SamplesLoss("sinkhorn", **kwargs)

    adam = torch.optim.Adam([x_new], lr=lr)

    for _ in range(n_iter):
        adam.zero_grad()
        loss = sample_loss(uniform, x_new, w_y, y)
        loss.backward()
        adam.step()
    return x_new.clone()


def recenter_from_target(w_y, y, lr=0.5, n_ts=5, n_iter=5, **kwargs):
    ts = torch.linspace(1 / n_ts, 1, n_ts, requires_grad=False)
    sample_loss = gl.SamplesLoss("sinkhorn", **kwargs)
    uniform = torch.ones_like(w_y, requires_grad=False)
    uniform /= uniform.sum()
    y_1 = y.clone()
    w_0 = w_y
    for t in ts:
        w_1 = (w_y * (-t + 1.) + t * uniform)
        y_0 = y_1.clone()
        y_0_clone = y_0.clone()
        y_1 = y_0.detach().requires_grad_(True)
        adam = torch.optim.Adam([y_1], lr=lr)
        for _ in range(n_iter):
            adam.zero_grad()
            loss = sample_loss(w_1, y_1, w_0, y_0_clone)
            loss.backward()
            adam.step()
        w_0 = w_1.detach()
    return y_1.clone()


def main():
    import time
    import matplotlib.pyplot as plt
    torch.random.manual_seed(0)
    n = 300
    x = torch.randn(n, 1)
    y, idx = torch.randn(n, 1).sort(0)
    w_y = 0.5 + torch.rand(n) * 0.5
    w_y /= w_y.sum()
    w_y[:100] = 0.
    print((w_y - 1 / n).abs().mean())
    print(y[100])
    tic = time.time()
    from_proposal = recenter_from_proposal(x, w_y, y, backend='tensorized').detach().numpy()
    print(time.time() - tic)

    tic = time.time()
    from_target = recenter_from_target(w_y, y, n_ts=3, n_iter=10, lr=0.25, backend='tensorized').detach().numpy()
    print(time.time() - tic)

    plt.hist(from_proposal.squeeze(), bins=30, alpha=0.5, label='from_proposal', density=True)
    plt.hist(from_target.squeeze(), bins=30, alpha=0.5, label='from_target', density=True)
    plt.hist(y.detach().squeeze().numpy().tolist(), weights=w_y.detach().numpy(), bins=30, alpha=0.5, label='initial',
             density=True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
