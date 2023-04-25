import pandas as pd
from bs4 import BeautifulSoup as bs
import emoji
import os

auto = "Telegram_auto"
boats = "Telegram_voda"
railways = "Telegram_railways"
files_auto = [file for file in os.listdir(auto)]  # получаем список файлов из папки про авто
files_boat = [file for file in os.listdir(boats)]  # полуаем список файлов из папки про лодки
files_railways = [file for file in os.listdir(railways)]  # полуаем список файлов из папки про поезда


def delete_emoji(txt):
    clean_text = ''.join(c for c in txt if c not in emoji.EMOJI_DATA)  # удаляю эмоджи из текста
    return clean_text


def boats_data():
    data = []
    for file in files_boat:
        # чтение html файла
        with open(f"Telegram_voda/{file}", 'r', encoding='UTF-8') as fp:
            soup = bs(fp, 'html.parser')

        for i in soup.find_all('div', class_='text'):
            if "https" in i.text:
                pass
            else:
                data.append(delete_emoji(i.text))
    data = [item.strip() for item in data]
    return data[2:]


def railways_data():
    data = []
    for file in files_railways:
        # чтение html файла
        with open(f"Telegram_railways/{file}", 'r', encoding='UTF-8') as fp:
            soup = bs(fp, 'html.parser')

        for i in soup.find_all('div', class_='text'):
            if "https" in i.text:
                pass
            else:
                data.append(delete_emoji(i.text))
    data = [item.strip() for item in data]
    return data[2:]


def auto_data():
    data = []
    for file in files_auto:
        # чтение html файла
        with open(f"Telegram_auto/{file}", 'r', encoding='UTF-8') as fp:
            soup = bs(fp, 'html.parser')

        for i in soup.find_all('div', class_='text'):
            if "https" in i.text:
                pass
            else:
                data.append(delete_emoji(i.text))
    data = [item.strip() for item in data]
    return data[2:]


def train_data():
    texts = []
    tags = []
    for i in auto_data():
        if "москв" in i.lower():
            texts.append(i)
            tags.append("транспорт Москвы")
        elif "cанк-" in i.lower():
            texts.append(i)
            tags.append("транспорт Санкт-Петербурга")
        else:
            texts.append(i)
            tags.append("Автомобильный транспорт")
    for i in boats_data():
        if "москв" in i.lower():
            texts.append(i)
            tags.append("транспорт Москвы")
        elif "cанк-" in i.lower():
            texts.append(i)
            tags.append("транспорт Санкт-Петербурга")
        else:
            texts.append(i)
            tags.append("Водный транспорт")
    for i in railways_data():
        if "москв" in i.lower():
            texts.append(i)
            tags.append("транспорт Москвы")
        elif "cанк-" in i.lower():
            texts.append(i)
            tags.append("транспорт Санкт-Петербурга")
        else:
            texts.append(i)
            tags.append("Железнодорожный транспорт")
    df = pd.DataFrame({'text': texts, 'tag': tags})
    df = df.dropna(how='any')
    df.to_csv("train_data.csv", index=False, lineterminator='')

if __name__=='__main__':
    train_data()
    df = pd.read_csv("train_data.csv")
    print(df.head())
