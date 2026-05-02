import lightning as L
import torch

class NextTokenPretraining(L.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        x, targets = batch
        x = torch.stack(x)
        targets = torch.stack(targets)

        # the model generates logits in the output vocab space for these
        # models, so we have to pop a cross-entropy loss on the end
        loss_class = torch.nn.CrossEntropyLoss()
        in_shape = x.shape
        reshape_sz = in_shape[0] * in_shape[1]
        logits = self.model(x)
        logits = torch.reshape(logits, (reshape_sz, logits.shape[-1]))
        loss = loss_class(logits, targets.flatten())
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return opt

