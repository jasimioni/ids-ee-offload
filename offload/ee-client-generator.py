#!/usr/bin/env python

import pika
import pandas as pd
import os
import uuid
import pickle
import logging
import torch
from datetime import datetime
import time
import json
import sys
import argparse
from datetime import datetime

rundir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rundir)
sys.path.append(os.path.dirname(rundir))

from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits
from calibration.temperature_scaling_2exits import ModelWithTemperature
from utils.functions import *

parser = argparse.ArgumentParser(description='Early Exits processor client.')

parser.add_argument('--mq-username', help='RabbitMQ username')
parser.add_argument('--mq-password', help='RabbitMQ password')
parser.add_argument('--mq-hostname', help='RabbitMQ hostname', required=True)
parser.add_argument('--mq-queue', help='RabbitMQ queue', default='ee-processor')
parser.add_argument('--device', help='PyTorch device', default='cpu')
parser.add_argument('--trained-network-file', help='Trainet network file', required=True)
parser.add_argument('--network', help='Network to use AlexNet | MobileNet', required=True)
parser.add_argument('--batch-size', help='Batch size', default=50, type=int)
parser.add_argument('--interval', help='Interval between requests (s)', default=0.1, type=int)
parser.add_argument('--debug', help='Enable debug messages', action='store_true')
parser.add_argument('--count', help='Number of repetitions', default=0, type=int)
parser.add_argument('--client-id', help='Client ID', default=os.getpid())

args = parser.parse_args()

log_level = logging.INFO if args.debug else logging.WARN
logging.basicConfig(level=log_level,
                    format='%(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)

device = torch.device(args.device)
if args.network == 'MobileNet':
    model = MobileNetV2WithExits().to(device)
else:
    model = AlexNetWithExits().to(device)

model_t = ModelWithTemperature(model, device=device)
model_t.load_state_dict(torch.load(args.trained_network_file))
model_t.model.eval()

class EEProcessorClient(object):
    def __init__(self):
        connection_params = { 'host': args.mq_hostname }
        if args.mq_username and args.mq_password:
            credentials = pika.PlainCredentials(args.mq_username, args.mq_password)
            connection_params['credentials'] = credentials

        self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(**connection_params))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, body):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=args.mq_queue,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=pickle.dumps(body))
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        
        try:
            return pickle.loads(self.response)
        except Exception as e:
            raise Exception(f"Failed to load: {e}")

eeprocessor = EEProcessorClient()

forever = False
counter = args.count
if args.count == 0:
    forever = True
    

X = torch.rand(args.batch_size, 1, 8, 8).to(device)
count = len(X)
bb1 = model_t.model.backbone[0](X)

total = 0

avg = []

while forever or counter > 0:
    counter -= 1
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    request = {
        'timestamp': now,
        'bb1': bb1,
        'client_id': args.client_id
    }
    
    offloaded = len(bb1)
    
    # logger.warning(f" [x] Requesting {now} {offloaded} being offloaded")

    start = time.time()
    response = eeprocessor.call(request)
    end = time.time()
    
    elapsed = end - start

    avg.append(elapsed)
    if len(avg) > 10:
        avg.pop(0)
        
    elapsed_avg = 1000 * sum(avg) / len(avg)
    per_item_avg = elapsed_avg / offloaded
    
    print(f"Elapsed: {1000 * elapsed:.2f} ms : {1000 * elapsed / offloaded:.2f} ms : {elapsed_avg:.2f} ms : {per_item_avg:.2f} ms")
    
    time.sleep(args.interval)
