import torch
import numpy as np
from tqdm import tqdm


class Training:
    """Class for training and evaluating the model."""

    def __init__(self, model, train_loader, valid_loader, device='cuda', epochs=25, lr=0.003):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_valid_loss = np.Inf

    def train_fn(self):
        """Training function for one epoch."""
        self.model.train()
        total_loss = 0.0

        for images, masks in tqdm(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits, loss = self.model(images, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def eval_fn(self):
        """Evaluation function for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(self.valid_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                logits, loss = self.model(images, masks)
                total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def run_training(self):
        """Main training loop."""
        for epoch in range(self.epochs):
            train_loss = self.train_fn()
            valid_loss = self.eval_fn()

            if valid_loss < self.best_valid_loss:
                torch.save(self.model.state_dict(), 'best_model.pt')
                print("Saved best model")
                self.best_valid_loss = valid_loss

            print(f"Epoch: {epoch + 1} Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f}")
