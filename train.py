import os
import glob
import torch
import numpy as np
import albumentations
from torch import nn
import json
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from functools import partial

import config
import dataset_class
import engine
from model import CaptchaSolver

def get_data():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
    image_files = [
        file for file in image_files
        if len(os.path.splitext(os.path.basename(file))[0].split()) <= 2
    ]

    targets_orig = [x.split("/")[-1][7:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]

    targets_flat = [c for clist in targets for c in clist]
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    return image_files, targets_enc, targets_orig, lbl_enc

def custom_collate_fn(batch):
    # Извлекаем изображения и метки из батча
    x = [sample['images'] for sample in batch]
    y = [sample['targets'] for sample in batch]

    # Преобразуем список меток в тензоры
    y = [torch.LongTensor(y_i) for y_i in y]
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)  # Добавляем паддинг

    # Преобразуем список изображений в тензор
    x_tensor = torch.stack(x)

    # Возвращаем словарь с тензорами
    return {
        "images": x_tensor,
        "targets": y_padded,
    }


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = []
    for j in x:
        if not fin or j != fin[-1]:
            fin.append(j)
    return ''.join(fin)



def decode_predictions(preds, encoder):
    # Перестановка осей для получения правильного формата
    preds = preds.permute(1, 0, 2)

    # Применение softmax и нахождение максимальных вероятностей
    preds = torch.softmax(preds, dim=2)
    preds = torch.argmax(preds, dim=2)  # Находим индексы с максимальной вероятностью
    
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k.item() - 1  # Преобразуем тензор в число
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k+1])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds

def run_training():
   
   image_files, targets_enc, targets_orig, lbl_enc = get_data()
   (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        train_targets_orig,
        test_targets_orig,
   ) = train_test_split(
       image_files, targets_enc, targets_orig, test_size=0.2, random_state=42
   )

   train_dataset = dataset_class.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
   )
   train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        collate_fn=custom_collate_fn
   )
   test_dataset = dataset_class.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
   test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=custom_collate_fn
   )


   model = CaptchaSolver(num_classes=len(lbl_enc.classes_))
   model.to(config.DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)

    # Для хранения ошибок и точности
   history = {"train_loss": [], "valid_loss": [], "accuracy": []}

    # Обучение модели
   for epoch in range(config.EPOCHS):
        # Шаг обучения
        train_loss = engine.train_step(model, train_loader, optimizer)
        
        # Шаг валидации
        valid_preds, valid_loss = engine.val_step(model, test_loader)
        valid_captcha_preds = []
        
        # Декодирование предсказаний
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_captcha_preds.extend(current_preds)
        
        combined = list(zip(test_targets_orig, valid_captcha_preds))
        print(combined[:10])
        
        # Удаление дубликатов из тестовых данных
        test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
        accuracy = accuracy_score(test_dup_rem, valid_captcha_preds)
        
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={valid_loss} Accuracy={accuracy}"
        )
        
        # Сохранение данных для построения графиков
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["accuracy"].append(accuracy)
        
        # Шаг планировщика
        scheduler.step(valid_loss)
    
    # Сохранение history в файл
   with open("training_history.json", "w") as f:
        json.dump(history, f)
        
   torch.save(model.state_dict(), "model_params.pth")

if __name__ == '__main__':
    run_training()
    

