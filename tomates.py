import os
import cv2
import shutil
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd


# Checar se o CUDA está disponível
print(f'{torch.cuda.is_available() = }')
print(f'{torch.cuda.device_count() = }')

dataset_path = '/home/carlos/Documentos/tomatinhos/yolo'

# Dicionário para armazenar informações das imagens e anotações
labels_dict = dict(
    img_path=[],
    xmin=[],
    xmax=[],
    ymin=[],
    ymax=[],
    img_w=[],
    img_h=[]
)

# Obter a lista de arquivos XML do diretório de anotações
xml_files = glob(f'{dataset_path}/annotations/*.xml')

for filename in xml_files:  # Processar todos os arquivos XML
    info = xet.parse(filename)
    root = info.getroot()

    objects = root.findall('object')

    img_name = root.find('filename').text
    img_path = os.path.join(dataset_path, 'Images', img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at path: {img_path}")
        continue

    height, width, _ = img.shape

    for member_object in objects:
        labels_info = member_object.find('bndbox')
        if labels_info is None:
            continue

        try:
            xmin = int(float(labels_info.find('xmin').text))
            xmax = int(float(labels_info.find('xmax').text))
            ymin = int(float(labels_info.find('ymin').text))
            ymax = int(float(labels_info.find('ymax').text))
        except (ValueError, AttributeError) as e:
            print(f"Error extracting bounding box in {filename}: {e}")
            continue

        # Salvar informações no dicionário
        labels_dict['img_path'].append(img_path)
        labels_dict['xmin'].append(xmin)
        labels_dict['xmax'].append(xmax)
        labels_dict['ymin'].append(ymin)
        labels_dict['ymax'].append(ymax)
        labels_dict['img_w'].append(width)
        labels_dict['img_h'].append(height)

# Dividir os dados em conjuntos de treinamento, validação e teste
data = pd.DataFrame(labels_dict)  # Se necessário
train, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)

# Criar pastas para YOLO
def make_split_folder_in_yolo_format(split_name, split_df):
    labels_path = os.path.join(dataset_path, split_name, 'labels')
    images_path = os.path.join(dataset_path, split_name, 'images')

    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    for _, row in split_df.iterrows():
        img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'a') as file:
            file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        destination_img_path = os.path.join(images_path, img_name + img_extension)
        shutil.copy(row['img_path'], destination_img_path)

    print(f"Created '{images_path}' and '{labels_path}' for {split_name} split.")

# Criar pastas para treino, validação e teste
make_split_folder_in_yolo_format("train", train)
make_split_folder_in_yolo_format("val", val)
make_split_folder_in_yolo_format("test", test)

# YAML de configuração do dataset
datasets_yaml = '''
path: /home/carlos/Documentos/tomatinhos/yolo

train: train/images
val: val/images
test: test/images

nc: 2
names: ['tomatinhos']
'''

# Salvar o arquivo de configuração
with open('datasets.yaml', 'w') as file:
    file.write(datasets_yaml)

# Carregar o modelo e iniciar o treinamento
model = YOLO('yolov8n.pt')  # Use o modelo pré-treinado que você preferir

# Iniciar o treinamento
model.train(
    data='/home/carlos/Documentos/tomatinhos/datasets.yaml',
    epochs=10,
    batch=16,
    device='cpu',  # Mude para 'cpu' se não houver GPU
    imgsz=320,
    cache=False,
    project='yolo_project',
    name='yolo_training',
    verbose=True
)

print("Treinamento concluído.")



def plot_training_metrics(log_file_path):
    """
    Lê o arquivo de log do YOLO e plota as métricas de treinamento.
    
    Parâmetros:
    log_file_path (str): Caminho para o arquivo 'results.csv' gerado pelo YOLO.
    """
    # Ler o arquivo CSV gerado pelo YOLO com as métricas de cada época
    results_df = pd.read_csv(log_file_path)

    # Plotar a perda (loss) ao longo das épocas
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(results_df['epoch'], results_df['train/box_loss'], label='Box Loss')
    plt.plot(results_df['epoch'], results_df['train/obj_loss'], label='Object Loss')
    plt.plot(results_df['epoch'], results_df['train/cls_loss'], label='Class Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda de Treinamento')
    plt.legend()

    # Plotar precisão e recall ao longo das épocas
    plt.subplot(2, 2, 2)
    plt.plot(results_df['epoch'], results_df['metrics/precision'], label='Precisão')
    plt.plot(results_df['epoch'], results_df['metrics/recall'], label='Recall')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.title('Precisão e Recall')
    plt.legend()

    # Plotar mAP (Mean Average Precision) 50 e mAP 50-95
    plt.subplot(2, 2, 3)
    plt.plot(results_df['epoch'], results_df['metrics/mAP_50'], label='mAP@50')
    plt.plot(results_df['epoch'], results_df['metrics/mAP_50-95'], label='mAP@50-95')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.title('mAP@50 e mAP@50-95')
    plt.legend()

    # Plotar a perda de validação ao longo das épocas
    plt.subplot(2, 2, 4)
    plt.plot(results_df['epoch'], results_df['val/box_loss'], label='Box Loss')
    plt.plot(results_df['epoch'], results_df['val/obj_loss'], label='Object Loss')
    plt.plot(results_df['epoch'], results_df['val/cls_loss'], label='Class Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda de Validação')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Especifique o caminho para o arquivo 'results.csv' gerado pelo YOLO
log_file_path = 'yolo_project/yolo_training/results.csv'

# Verifique se o arquivo existe antes de tentar plotar
if os.path.exists(log_file_path):
    plot_training_metrics(log_file_path)
else:
    print(f"Erro: Arquivo de resultados não encontrado no caminho: {log_file_path}")


# Após o treinamento, salvar o modelo manualmente em um local personalizado
model.save('tomates.pt')
