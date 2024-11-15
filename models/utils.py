class EarlyStopper:
    def __init__(self, patience=10, threshold=1e-2, min_delta=1e-3):
        self.patience = patience
        self.threshold = threshold
        self.min_delta = min_delta
        self.counter = 0
        self.loss_min = float('inf')

    def early_stop(self, loss):
        if loss < self.threshold:
            return True
        if loss < self.loss_min:
            self.loss_min = loss
            self.counter = 0
        elif loss > (self.loss_min + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False