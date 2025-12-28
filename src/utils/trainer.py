import torch
import torch.nn as nn
from tqdm import tqdm
from .config import get_device

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader=None):
        self.device = get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # History
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', unit='batch')
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self):
        if not self.test_loader:
            return 0.0, 0.0
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)
        
        print(f" > Validation: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def train(self, epochs):
        print(f"Starting training on {self.device} for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            if self.test_loader:
                self.evaluate()
        print("Training finished.")
        return {
            'train_loss': self.train_losses,
            'test_loss': self.test_losses,
            'test_acc': self.test_accuracies
        }
