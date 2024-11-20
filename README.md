# CAPTCHA Recognition Project

В данном проекте была реализована нейронная сеть, способная решать Captcha с Accuracy > 85%.

Для прямого запуска скрипта: 

```bash
python captcha_gen_default.py

## 1. Prepare CAPTCHAs

Данные для обучения и валидации модели находятся в  `<current_dir>/280x70/`. Названия файлов - решение капчи.

Рекомендованный размер `IMAGE_WIDTH = 280, IMAGE_HEIGHT = 80`. Названия файлов формата `*.jpg.`

```bash
python captcha_gen_default.py
