
import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class PLU(nn.Module):
    """Probability Linear Unit"""

    def __init__(self):
        super(PLU, self).__init__()

    def forward(self, x):
        z = torch.clamp(x, 0, 1)
        return z


class AutoDropout(nn.Module):
    def __init__(self, dp=0., requires_grad=False, straight_thru_grad=False):

        super(AutoDropout, self).__init__()

        # We transform the dropout rate to keep rate
        p = 1 - dp
        p = torch.tensor(p)

        self.plu = PLU()

        if requires_grad:
            p = nn.Parameter(p)
            self.register_parameter("p", p)
        else:
            self.register_buffer("p", p)

        self.straight_thru_grad = straight_thru_grad

    def forward(self, x):
        bs, shape = x.shape[0], x.shape[1:]

        # We make sure p is a probability
        p = self.plu(self.p)

        ps = p.expand(shape)

        m = Bernoulli(ps).sample((1,)).squeeze(0)

        if self.straight_thru_grad:
            m = ps + (m - ps).detach()

        # Element wise multiplication
        z = x * m

        return z

    def extra_repr(self):
        return 'p={}'.format(
            self.p.item()
        )


class DropLinear(nn.Module):
    def __init__(self, in_features, out_features, dp=0., bias=True, requires_grad=False, straight_thru_grad=False):
        super(DropLinear, self).__init__()

        self.dp = AutoDropout(dp=dp, requires_grad=requires_grad,
                              straight_thru_grad=straight_thru_grad)
        self.W = nn.Linear(in_features=in_features,
                           out_features=out_features, bias=bias)
        self.W.weight.data = self.W.weight.data / self.W.weight.data.norm() * (1-dp)

    def forward(self, x):
        z = self.W(x)
        z = self.dp(z)
        return z
