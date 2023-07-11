from download_doodles import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np


class DoodleClassifier(pl.LightningModule):
    # def __init__(self, classes = ["The Eiffel Tower", "alarm clock", "axe", "banana"]):
    #     super().__init__()
    #     self.classes = classes
    #     self.conv1 = nn.Conv2d(1, 32, 5, padding = 2)
    #     self.conv2 = nn.Conv2d(32, 64, 5, padding = 2)
    #     self.pool = nn.MaxPool2d(2, 1)
    #     self.fc1 = nn.Linear(64 * 26 * 26, 128)
    #     self.fc2 = nn.Linear(128, len(classes))

    # def forward(self, X):
    #     if len(X.size()) == 3:
    #         X = X.unsqueeze(1)
    #     X = self.pool(F.relu(self.conv1(X)))
    #     X = self.pool(F.relu(self.conv2(X)))
    #     X = X.view(-1, 64 * 26 * 26)
    #     X = F.relu(self.fc1(X))
    #     X = self.fc2(X)
    #     X = torch.sigmoid(X)
    #     return X
    def __init__(self, classes = ["The Eiffel Tower", "alarm clock", "axe", "banana"]):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(28 * 28, 512)
        # self.fc_out = nn.Linear(512, len(self.hparams.classes))
        self.fc_out = nn.Linear(512, 19)
        self.fcs = [nn.Linear(512, 512) for _ in range(5)]
        self.seq = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fcs[0],
            nn.ReLU(),
            self.fcs[1],
            nn.ReLU(),
            self.fcs[2],
            nn.ReLU(),
            self.fcs[3],
            nn.ReLU(),
            self.fcs[4],
            nn.ReLU(),
            self.fc_out,
            # nn.Sigmoid(),
        )


    def forward(self, X):
        return self.seq(X.view(X.size(0), 28 * 28))
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        # Calculate accuracy
        acc = (y_hat.argmax(dim = 1) == y.argmax(dim = 1)).float().mean()
        self.log("train_acc", acc, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar = True)
        # Calculate accuracy
        acc = (y_hat.argmax(dim = 1) == y.argmax(dim = 1)).float().mean()
        self.log("val_acc", acc, prog_bar = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.binary_cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        # Calculate accuracy
        acc = (y_hat.argmax(dim = 1) == y.argmax(dim = 1)).float().mean()
        self.log("test_acc", acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        return [optimizer], [optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.5)]
    
    def predict(self, X, classes = None):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        y_hat = self(X)
        # Remove irrelevant classes
        if classes is not None:
            indices = [self.hparams.classes.index(c) for c in classes]
            y_hat = y_hat[:, indices]
        y_hat = torch.softmax(y_hat, dim = 1)
        return y_hat.argmax(dim = 1).numpy()

    def predict_probs(self, X, classes = None):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        y_hat = self(X)
        # Remove irrelevant classes
        if classes is not None:
            indices = [self.hparams.classes.index(c) for c in classes]
            y_hat = y_hat[:, indices]
        y_hat = torch.softmax(y_hat, dim = 1)
        return y_hat.detach().numpy()


if __name__ == "__main__":
    train_data, val_data, test_data = load_data(percent_data = 0.5)
    model = DoodleClassifier()
    trainer = pl.Trainer(max_epochs = 1,
                         accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                         devices = 1,
                         val_check_interval = 0.1)
    trainer.fit(model, train_dataloaders = train_data, val_dataloaders = val_data)
    trainer.test(model, dataloaders = test_data)