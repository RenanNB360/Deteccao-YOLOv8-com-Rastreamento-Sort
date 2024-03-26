# Detecção de Objetos com YOLOv8

# Importando os arquivos
import math
import time
import numpy as np
import cv2
from ultralytics import YOLO
import pygame
from modulo import Sort

# carregando o video
cap = cv2.VideoCapture('gado.mp4')

# Definindo o idioma
language = 'en'

# Carregando o modelo de detecção YOLO (yolov8n.pt)
modelo = YOLO('yolov8n.pt')

# Nome das classes
classNames = ['Dog', 'Koala', 'Zebra', 'pig', 'antelope', 'badger', 'bat', 'bear', 'bison', 'cat', 'chimpanzee', 
              'cow', 'coyote', 'deer', 'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fox', 'goat', 'goldfish', 
              'goose', 'gorilla', 'hamster', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
              'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'pigeon',
              'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'peguin', 'pelecaniformes', 'porcupine', 
              'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 
              'starfish', 'swain', 'tiger', 'turkey', 'turtle', 'undetected', 'whale', 'whale-shark', 'wolf', 'woodpecker']


tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)


limits = [400,400,700,700]
limitT = [400,400,700,700]
totalCount = []


while True:
    success, img = cap.read()
    resultados = modelo(img, stream= True)
    detections = np.empty((0,5))
    for r in resultados:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)
            W, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
    resultadosTracker = tracker.update(detections)
    for resultado in resultadosTracker:
        x1, y1, x2, y2, id = resultado
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(resultado)
    try:
        cv2.imshow('Image', img)
    except:
        break

    cv2.waitKey(1)