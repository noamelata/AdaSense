import torch

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, etaA, classes=None):
    with torch.no_grad():
        #setup iteration variables
        at = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        x = at.sqrt() * H_funcs.H_pinv(y_0).view(*x.size()) + (1 - at).sqrt() * x
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        xt_next = x

        #iterate over the timesteps
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xt_next
            if classes is None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            xt_next_null = at_next.sqrt() * x0_t + ((1 - etaA) ** 0.5) * (1 - at_next).sqrt() * et + etaA * (1 - at_next).sqrt() * torch.randn_like(x0_t)
            xt_next = xt_next_null - H_funcs.H_pinv(H_funcs.H(xt_next_null.reshape(xt_next_null.shape[0], -1))).reshape(xt_next_null.shape) + \
                      H_funcs.H_pinv(at_next.sqrt()[0, 0, 0, 0] * y_0 + H_funcs.H(((1 - at_next).sqrt() * torch.randn_like(xt_next_null)).reshape(xt_next_null.shape[0], -1))).reshape(xt_next_null.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds