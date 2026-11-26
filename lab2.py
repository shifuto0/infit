
import cv2
import numpy as np
from datetime import datetime

name = input("Введите имя файла: ")
path = "data/input/" + name

try:
    # Чтение изображения
    with open(path, "rb") as file:
        data = file.read()

    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise OSError("Не удалось распознать изображение")

    # Гамма-коррекция
    gamma = 0.5
    corrected = (255 * (image / 255.0) ** gamma).astype(np.uint8)

    # Детектор границ Кэнни
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    result = cv2.Canny(blurred, 30, 60)

    # Сохранение результатов
    if not cv2.imwrite("data/output/gamma.jpg", corrected):
        raise OSError("Ошибка сохранения изображения")

    if not cv2.imwrite("data/output/result.jpg", result):
        raise OSError("Ошибка сохранения изображения")

    print("Изображение успешно обработано")
    status = "Успешно"

except PermissionError:
    print(f"Ошибка: Нет прав на чтение файла {name}. "
          "Обратитесь к администратору")
    status = "Ошибка доступа"

except FileNotFoundError:
    print("Ошибка: Файл не найден")
    status = "Файл не найден"

except (OSError, cv2.error) as error:
    print("Ошибка обработки:", error)
    status = "Ошибка обработки"

# Запись в журнал
time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

with open("logs/history.txt", "a", encoding="UTF-8") as log:
    log.write(f"[{time}] Файл: {name} | "
              f"Фильтр: Гамма-коррекция + Кэнни | "
              f"Статус: {status}\n")
