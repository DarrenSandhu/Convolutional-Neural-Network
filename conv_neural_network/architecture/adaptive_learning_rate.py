import torch
import math
import time

class AdaptiveLearningRateDecay:
    def __init__(self, 
                 initial_lr=0.01, 
                 decay_type='exponential', 
                 patience=5, 
                 factor=0.5, 
                 min_lr=1e-5):
        """
        Adaptive learning rate decay with multiple strategies
        
        Args:
            initial_lr (float): Starting learning rate
            decay_type (str): Decay strategy ('exponential', 'step', 'plateau')
            patience (int): Number of epochs to wait before reducing learning rate
            factor (float): Multiplicative factor for learning rate reduction
            min_lr (float): Minimum learning rate threshold
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_type = decay_type
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        # Tracking variables
        self.epochs_no_improve = 0
        self.best_loss = float('inf')
    
    def step(self, current_loss, epoch):
        """
        Update learning rate based on selected decay strategy
        
        Args:
            current_loss (float): Current epoch's average loss
            epoch (int): Current training epoch
        
        Returns:
            float: Updated learning rate
        """
        if self.decay_type == 'exponential':
            # Exponential decay with adaptive rate
            decay_rate = max(0.95, 1 - (epoch * 0.01))
            self.current_lr = max(self.min_lr, self.initial_lr * (decay_rate ** epoch))
        
        elif self.decay_type == 'step':
            # Step decay: reduce learning rate every fixed number of epochs
            if epoch > 0 and epoch % self.patience == 0:
                self.current_lr = max(self.min_lr, self.current_lr * self.factor)
        
        elif self.decay_type == 'plateau':
            # Plateau-based learning rate reduction
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                
                if self.epochs_no_improve >= self.patience:
                    self.current_lr = max(self.min_lr, self.current_lr * self.factor)
                    self.epochs_no_improve = 0
        
        return self.current_lr