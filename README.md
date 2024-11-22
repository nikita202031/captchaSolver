# CAPTCHA Recognition Project

В данном проекте реализуется архитектура нейронной сети, способная решать Captcha с Accuracy > 85%.
![3a3a3k](https://github.com/user-attachments/assets/9e7b2807-01b1-4546-a987-aa629154a222)
![3a3zm3](https://github.com/user-attachments/assets/ebccf3b0-701e-41da-bf4b-18bf9d74f29d)
![8wsuqp](https://github.com/user-attachments/assets/6e6ba0d9-951e-44f8-a057-73040c939660)



Для прямого запуска скрипта: 

```bash
python solve_captcha.py
```
# Описание проекта

## 1. Подготовка данных

Данные для обучения и валидации модели находятся в  `<current_dir>/280x70/`. Названия файлов - решение капчи.

Рекомендованный размер `IMAGE_WIDTH = 280, IMAGE_HEIGHT = 80`. Названия файлов формата `*.jpg.`

## 2. Обучение модели

Архитектура модели хранится в `model.py`. Это простая CNN-GRU модель, на выходе дающая логиты по всем классам (классы представляют собой все символы, содержащиеся в капче). Функция потерь - CTCLoss. 

## 3. Отображение результатов

После обучения получаем 2 файла - `trainig_history.json` и `model_params.pth`. Отображение резульатов обучения:

```bash
python train_results.py
```
![image](https://github.com/user-attachments/assets/0600f9e8-3bf2-4ee2-9dae-2cbc3fd0f91a)

