import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram import executor

import cv2
import numpy as np
from keras.models import load_model


# Загрузка обученной модели
model_path = './model_x-ray_my_best.h5'
model = load_model(model_path)

load_dotenv()

# Инициализация бота и хранилища состояний
bot = Bot(os.getenv('TOKEN'))
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

@dp.message_handler(commands=['start'])
async def command_start(message: types.Message):
    await message.answer(f'Здравствуйте {message.from_user.first_name}! \nзагрузите Ваш снимок рентгена')


@dp.message_handler(content_types=['photo'])
async def check_photo(message: types.Message):
    await message.answer('Снимок обрабатывается...')              
    fail_info = await bot.get_file(message.photo[-1].file_id)
    await message.photo[-1].download(fail_info.file_path.split('photos/')[1])
    path = fail_info.file_path.replace('photos/', './')
    # Обработка фото
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (24, 42))
    image = image.astype('float32') / 255.0  # нормируем
    image = np.array(image)  # Перевод в numpy-массив
    image = np.expand_dims(image, axis=0)  # Добавляем размерность
    # Предсказание моделью и выдача результата пользователю
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction[0])
    if predicted_class == 0:
        variant = f'Поздравляем {message.from_user.first_name} у Вас нет пневмонии.\nЖелаем здоровья!!!'
    else:
        variant = f'{message.from_user.first_name} мы можем с определенной долей вероятности ошибатся,\nно есть предположение что у Вас пневмония.\nСоветуем обратится к доктору'
    await message.answer(f'{variant}')
    # Удаление обработанного фото
    os.remove(path)

# Если вдруг поступило на обработку не фото
@dp.message_handler()
async def answer_nothingcommand_start(message: types.Message):
    await message.reply(f'{message.from_user.first_name} видимо Вы загрузили в чат не изображение рентген снимка')


    # Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)