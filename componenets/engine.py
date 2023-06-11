import torch
import torch.optim as optim

from componenets.model import Model


class trainer():
    def __init__(self, scaler, device, lrate, num_nodes, input_dim, output_dim, channels,
                 grad_norm, dynamic, lag, horizon, supports, memory_size,hidden_size,lr_decay_step,lr_decay_rate,is_lr,proxies):
        self.model = Model(device=device, num_nodes=num_nodes, input_dim=input_dim, output_dim=output_dim,
                           channels=channels, dynamic=dynamic, lag=lag, horizon=horizon, supports=supports,
                           memory_size=memory_size,hidden_size=hidden_size,is_lr=is_lr,proxies=proxies)
        self.model.to(device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lrate)
        self.loss = torch.nn.SmoothL1Loss()
        self.scaler = scaler
        self.clip = 5
        self.grad_norm = grad_norm
        self.dynamic = dynamic
        self.device = device
        self.adj_mx = supports[0]
        self.is_lr=is_lr

        if self.is_lr:
            lr_decay_steps = [int(i) for i in list(lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,milestones=lr_decay_steps,gamma=lr_decay_rate)

    def train(self, input, real_val, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input,self.adj_mx)
        label = self.scaler.inverse_transform(real_val)
        output = self.scaler.inverse_transform(output)


        prediction = self.loss(output, label)

        loss = prediction
        loss.backward()

        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return prediction.item()

    def eval(self, input, real_val):
        output = self.model(input,self.adj_mx)
        label = self.scaler.inverse_transform(real_val)
        output = self.scaler.inverse_transform(output)

        loss = self.loss(output, label)
        return loss.item()
