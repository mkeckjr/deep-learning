import lightning as L
import torch

class NextTokenPretraining(L.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        x, targets = batch

        # the model generates logits in the output vocab space for these
        # models, so we have to pop a cross-entropy loss on the end
        loss_class = torch.nn.CrossEntropyLoss()
        logits = self.model(x)
        loss = loss_class(logits, target)
        return loss

    def configure_optimizers(self):
        opt = torch.nn.Adam(self.parameters, lr=1e-3))
        return opt
