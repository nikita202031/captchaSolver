# Загрузка модели на GPU
import torch
import os
import glob
import model
import config
import train
import gradio as gr
from torchvision import transforms
from sklearn import preprocessing

def captcha_prediction():
    _, _, _, lbl_enc = train.get_data()


    def predict_captcha(image):
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            #transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(),  # Нормализация
        ])
        
        # Обработка изображения
        image = transform(image).to(config.DEVICE)
        nnCaptcha = model.CaptchaSolver()
        nnCaptcha.load_state_dict(torch.load("model_params.pth", map_location=config.DEVICE))
        nnCaptcha.to(config.DEVICE)
        nnCaptcha.eval()  # Установка модели в режим валидации
        with torch.no_grad():
            # Передаем изображение в модель
            batch_preds = nnCaptcha(images=image)  # Используем ключевое слово `images` как в train/val
            # Декодируем предсказания
            decoded_preds = train.decode_predictions(batch_preds, lbl_enc)
        
        return decoded_preds[0]  # Возвращаем первое предсказание

    # Интерфейс Gradio
    interface = gr.Interface(
        fn=predict_captcha,
        inputs=gr.Image(type="pil", label="Загрузите изображение капчи"),
        outputs=gr.Textbox(label="Распознанная капча"),
        title="Распознавание капчи",
        description="Загрузите изображение капчи, чтобы получить предсказание."
    )

    # Запуск интерфейса
    interface.launch()

if __name__ == '__main__':
    captcha_prediction()
    
