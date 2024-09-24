from torch import nn

class LogReg(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        self.linear = nn.Linear(nfeat, 1)
        #self.softmax = nn.functional.log_softmax()

    def forward(self, x):
        return self.linear(x)
        #return nn.functional.log_softmax(self.linear(x), dim=1)
