#!/usr/bin/env python
from datetime import datetime
import sys
import time
import torch
import pickle
import pika
import socket
import argparse
import os

rundir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rundir)
sys.path.append(os.path.dirname(rundir))

from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits
from calibration.temperature_scaling_2exits import ModelWithTemperature

parser = argparse.ArgumentParser(description='Early Exits processor server.')

parser.add_argument('--mq-username', help='RabbitMQ username')
parser.add_argument('--mq-password', help='RabbitMQ password')
parser.add_argument('--mq-hostname', help='RabbitMQ hostname', required=True)
parser.add_argument('--mq-queue', help='RabbitMQ queue', default='ee-processor')
parser.add_argument('--device', help='PyTorch device', default='cpu')
parser.add_argument('--trained-network-file', help='Trainet network file', required=True)
parser.add_argument('--network', help='Network to use AlexNet | MobileNet', required=True)

args = parser.parse_args()

device = torch.device(args.device)
if args.network == 'MobileNet':
    model = MobileNetV2WithExits().to(device)
else:
    model = AlexNetWithExits().to(device)
    
model_t = ModelWithTemperature(model, device=device)
model_t.load_state_dict(torch.load(args.trained_network_file, map_location=device))

connection_params = { 'host': args.mq_hostname }
if args.mq_username and args.mq_password:
    credentials = pika.PlainCredentials(args.mq_username, args.mq_password)
    connection_params['credentials'] = credentials

connection = pika.BlockingConnection(
    pika.ConnectionParameters(**connection_params))

channel = connection.channel()
channel.queue_declare(queue=args.mq_queue)

def on_request(ch, method, props, body):
    start = time.time()
    time_records = {'start': start}
    print(f" {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

    response = {}
    response['input_size'] = sys.getsizeof(body)
    body = pickle.loads(body)

    # sample = body['sample'].to(device)
    bb1 = body['bb1'].to(device)

    bb2 = model_t.model.backbone[1](bb1)
    e2 = model_t.model.exits[1](bb2)
    y_pred = model_t.temperature_scale(1, e2)

    response['output'] = y_pred.to(torch.device('cpu'))
    response['hostname'] = socket.gethostname()

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=pickle.dumps(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=args.mq_queue, on_message_callback=on_request)

print("Waiting for RPC requests")
channel.start_consuming()
