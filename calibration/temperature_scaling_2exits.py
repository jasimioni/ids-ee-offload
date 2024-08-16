import torch
from torch import nn, optim
from torch.nn import functional as F
import sys

# https://www.kaggle.com/code/anonamename/temperature-scaling
# https://github.com/ondrejbohdal/meta-calibration/blob/main/temperature_scaling.py
# https://github.com/Jonathan-Pearce/calibration_library/blob/master/recalibration.py



class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device=torch.device('cpu'), max_iter=50, epochs=6):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.model.eval()
        self.device = device
        self.max_iter = max_iter
        self.epochs = epochs
        
        # print(f"Created with {self.epochs} epochs and {self.max_iter} max iterations.")
        
        # Initialize the temperature parameter as a list of parameters
        
        self.temperature = nn.ParameterList([
            nn.Parameter(torch.ones(1).to(self.device) * 1.5),
            nn.Parameter(torch.ones(1).to(self.device) * 1.5),
        ])
        
    def forward(self, input):
        return [ self.forward_exit(0, input),
                 self.forward_exit(1, input) ]

    def forward_exit(self, exit, input):
        if self.model.measurement_mode:
            logits, avg_bb_time, avg_exit_time = self.model.forward_exit(exit, input)
            return self.temperature_scale(exit, logits), avg_bb_time, avg_exit_time
        else:
            logits = self.model.forward_exit(exit, input)
            return self.temperature_scale(exit, logits)

    def temperature_scale(self, exit, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature[exit].unsqueeze(1).expand(logits.size(0), logits.size(1))

        return logits / temperature

    # This function probably should live outside of this class, but whatever

    def set_temperature(self, valid_loader):
        self.set_temperature_exit(0, valid_loader)
        self.set_temperature_exit(1, valid_loader)
        return self

    def set_temperature_exit(self, exit, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss(n_bins=10).to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model.forward_exit(exit, input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature[exit]], lr=0.01, max_iter=self.max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(exit, logits), labels)
            loss.backward()
            return loss

        best = {
            'ece': 1,
            'temperature': None
        }
        
        for _ in range(self.epochs):
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(exit, logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(exit, logits), labels).item()
            
            if after_temperature_ece < best['ece']:
                best['ece'] = after_temperature_ece
                best['temperature'] = self.temperature[exit].item()
                
            print(f'Epoch {_:02d} exit {exit}: Temperature: {self.temperature[exit].item():.5f}')
            print(f'                 After temperature NLL: {after_temperature_nll:.5f}, ECE: {after_temperature_ece:.5f}')
            print(f'                 Best temperature so far: {best["temperature"]:.5f}, ECE: {best["ece"]:.5f}')

        self.temperature[exit] = nn.Parameter(torch.ones(1).to(self.device) * best['temperature'])
        print(f'Before temperature for {exit} - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')
        print(f"Best temperature for {exit}: {self.temperature[exit].item():.5f}, ece: {best['ece']:.5f}")

        return self
    
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(.5, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
