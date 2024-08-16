import torch.nn as nn
import torch
import time

# https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV2.py

def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
#            nn.BatchNorm2d(ch_in, track_running_stats=False),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
#            nn.BatchNorm2d(ch_out, track_running_stats=False),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
#            nn.BatchNorm2d(ch_out, track_running_stats=False),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super().__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, ch_in=1, num_classes=2):
        super().__init__()

        configs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        
        self.adapt = nn.AdaptiveAvgPool2d((48, 48))
        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        backbone = []

        input_channel = 32
        for t, c, n, s in configs:
            for i in range(n):
                stride = s if i == 0 else 1
                backbone.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.backbone = nn.Sequential(*backbone)

        self.exit = nn.Sequential(
            conv1x1(input_channel, 1280),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )
        )

        self.measurement_mode = False

    def forward(self, x):
        st = time.process_time()
        x = self.adapt(x)
        x = self.stem_conv(x)
        x = self.backbone(x)
        x = self.exit(x)
        ed = time.process_time()

        print(f'{ed} - {st}: {ed - st}')

        if self.measurement_mode:
            return x, ed - st

        return x

    def set_measurement_mode(self, mode=True):
        if mode:
            pass 
            # self.eval()
        self.measurement_mode = mode

class MobileNetV2WithExits(nn.Module):
    def __init__(self, ch_in=1, num_classes=2, exit_loss_weights=[1, 1]):
        super().__init__()

        exits = []

        configs_ee=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
        ]

        configs = [
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        
        bb1 = [
            nn.AdaptiveAvgPool2d((48, 48)),
            conv3x3(ch_in, 32, stride=2)
        ]

        input_channel = 32
        for t, c, n, s in configs_ee:
            for i in range(n):
                stride = s if i == 0 else 1
                bb1.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        exits.append(nn.Sequential(
            conv1x1(input_channel, 1280),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )
        ))     

        bb2 = []

        for t, c, n, s in configs:
            for i in range(n):
                stride = s if i == 0 else 1
                bb2.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c
        
        exits.append(nn.Sequential(
            conv1x1(input_channel, 1280),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )
        ))

        self.backbone = nn.Sequential(
            nn.Sequential(*bb1),
            nn.Sequential(*bb2)
        )
        
        self.exits    = nn.Sequential(*exits)        

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
