import torch.nn as nn
import torch
import time

class AlexNetLayers():
    def __init__(self, num_classes=2):
        self.backbone = [          
            nn.Sequential(
                nn.AdaptiveAvgPool2d((48, 48)),
                nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2)
                )
            ),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()),
                
                nn.Sequential(
                    nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()),

                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()),

                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()),
                
            ),
        ]

        self.exits = [
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1600, num_classes)),

            nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(6400, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, num_classes))
        ]

class AlexNetEE(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = AlexNetLayers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:1])
        self.exit     = layers.exits[0]

        self.measurement_mode = False

    def forward(self, x):
        return self.exit(self.backbone(x))

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = AlexNetLayers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone)
        self.exit     = layers.exits[1]

        self.measurement_mode = False

    def forward(self, x):
        if self.measurement_mode:
            st = time.process_time()
            x = self.exit(self.backbone(x))
            ed = time.process_time()
            return x, ed - st
        return self.exit(self.backbone(x))        

    def set_measurement_mode(self, mode=True):
        if mode:
            pass 
            # self.eval()
        self.measurement_mode = mode

class AlexNetWithExits(nn.Module):
    def __init__(self, num_classes=2, exit_loss_weights=[1, 1]):
        super().__init__()

        layers = AlexNetLayers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone)
        self.exits    = nn.Sequential(*layers.exits)

        self.exit_threshold = torch.tensor([0.5, 0.7], dtype=torch.float32)
        self.fast_inference_mode = False
        self.measurement_mode = False
        self.exit_loss_weights = exit_loss_weights

    def exit_criterion(self, ee_n, x):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            nc = torch.max(pk)
            return nc > self.exit_threshold[ee_n]

    def _forward_all_exits(self, x):
        results = []
        for bb, ee in zip(self.backbone, self.exits):
            if self.measurement_mode:
                st = time.process_time()
                x = bb(x)
                im = time.process_time()
                res = ee(x)
                ed = time.process_time()
                results.append([ res, im - st, ed - im ])
            else:
                x = bb(x)
                results.append(ee(x))

        return results

    def forward_exit(self, exit, x):
        timers = [ time.process_time()]
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            timers.append(time.process_time())
            if i == exit:
                output = self.exits[i](x)
                timers.append(time.process_time())
                if self.measurement_mode:
                    return [ output, timers[-2] - timers[-3], timers[-1] - timers[-2] ]
                return output

    def forward(self, x):
        if self.fast_inference_mode:
            for ee_n, (bb, ee) in enumerate(zip(self.backbone, self.exits)):
                x = bb(x)
                res = ee(x)
                if self.exit_criterion(ee_n, res):
                    return [res, 'ee' + ee_n]
            return [res, 'main']

        return self._forward_all_exits(x)

    def exits_certainty(self, x):
        results = []
        for ee_n, (bb, ee) in enumerate(zip(self.backbone, self.exits)):
            x = bb(x)
            res = ee(x)
            certainty, predicted = torch.max(nn.functional.softmax(res, dim=-1), 1)
            results.append([ certainty.item(), predicted.item() ])
        return results

    def set_fast_inference_mode(self, mode=True):
        if mode:
            pass
            # self.eval()
        self.fast_inference_mode = mode

    def set_measurement_mode(self, mode=True):
        if mode:
            pass 
            # self.eval()
        self.measurement_mode = mode
