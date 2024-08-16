#!/usr/bin/env python3

# TODO - Add direct download links

models = [
    ( 'AlexNet_calibrated.pth', 'https://drive.google.com/file/d/1uCDKoIx7woGinxInA4xGyMWvcwMpKnA7/view?usp=drive_link' ),
    ( 'AlexNet.pth', 'https://drive.google.com/file/d/15r9b1HlWU_v6qexesVxTjVnzrDZlmpH8/view?usp=drive_link' ),
    ( 'AlexNetWithExits_calibrated.pth', 'https://drive.google.com/file/d/18hXOmjgmYNbYNIrKVVAT9SB9WNdVFUfU/view?usp=drive_link' ),
    ( 'AlexNetWithExits.pth', 'https://drive.google.com/file/d/1ckwMqDlcLWpSqlaUi1eR7nmCY8bEY6L4/view?usp=drive_link' ),
    ( 'MobileNetV2.pth', 'https://drive.google.com/file/d/1GCb85UcBNhENtjvXtpYYPNf-f08NLXiN/view?usp=drive_link' ),
    ( 'MobileNetV2WithExits_calibrated.pth', 'https://drive.google.com/file/d/1thR1A3tI_9jKAhtCjAUY1fymvHx19SuD/view?usp=drive_link' ),
    ( 'MobileNetV2WithExits.pth', 'https://drive.google.com/file/d/1eBhCJSr7GWEVlIAHvCFw3j11Vyv3Y38X/view?usp=drive_link' ),
]

for filename, url in models:
    print(f'Access {url} to download {filename}')