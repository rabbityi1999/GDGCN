import torch.optim as optim
from GDGCN import *
import util

criterion = nn.SmoothL1Loss()
class trainer():
    def __init__(self, args,scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device):
        if args.model_name=='gdgcn':
            self.model = GDGCN(device, num_nodes, dropout, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid,
                               skip_channels=nhid * 8, layers=args.layers,temporal_mode = args.temporal_mode, ablation_mode = args.ablation_mode)
        else:
            print("model name error!")
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if args.loss == 'huber':
            print("It's huber loss!")
            self.loss = util.masked_huber
        else:
            self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, time_i):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, time_i)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val, time_i):
        self.model.eval()
        output = self.model(input, time_i)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
