import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

# Указываем  путь к fallback-файлу config.json
CONFIG_FILE = "C:\\Temp\\Special\\config.json"
TEXT_FILE = "instruction.txt"
OUTPUT_JSON = "dataset.json"


# Загрузка API-ключа из .env или fallback-файла config.json и настройка Gemini.
def configure_api():
    load_dotenv()
    api_key = os.getenv("PVY_GEMINI_API_KEY")
    if not api_key:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)
                api_key = config.get("PVY_GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API ключ не найден ни в .env, ни в config.json!")
    genai.configure(api_key=api_key)

# Разбиение текста на логические блоки (например, по пустым строкам)
def split_text_blocks(text):
    blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    return blocks

# Отправка блока в Gemini для генерации смысла (ответа)
def analyze_block(block, model):
    prompt = (f"Прочитай следующий фрагмент и сформулируй краткий смысл, "
              f"как если бы ты учил другого человека:\n\n{block}")
    while True:
        # обращение с одним и тем же промптом делаем в потенциальном
        # цикле, ввиду того, что может быть отказ по ошибке ограничения
        # кол-ва обращений
        try:
            response = model.generate_content(prompt)
            break  # Если успешен — выходим из цикла
        except Exception as e:
            # если от Gemini придёт ошибка №429 (ограничение по кол-ву обращений,
            # то надо вынуть из неё данные о том, какое время надо подождать до
            # следующего обращения
            if "429" in str(e):
                # Попытка найти retry_delay в details
                delay_seconds = None
                try:
                    retry_info = next((d for d in e.details if "retry_delay" in str(d)), None)
                    if retry_info:
                        retry_info_str = str(retry_info).replace("\n", " ").replace("  ", " ")

                        # находим индекс подстроки "seconds:" и вынимаем за ним
                        # цифры требуемой моделью задержки до повторного обращения
                        index = retry_info_str.find("seconds:")
                        if index != -1:
                            # Получаем всё, что идёт после "seconds:"
                            after_seconds = retry_info_str[index + len("seconds:"):]

                            # Очищаем ведущие пробелы
                            after_seconds = after_seconds.lstrip()

                            # Собираем цифры
                            number_str = ""
                            for ch in after_seconds:
                                if ch.isdigit():
                                    number_str += ch
                                else:
                                    break

                            # Преобразуем в int, если найдено
                            if number_str:
                                delay_seconds = int(number_str)

                except Exception:
                    pass
                # задаём найденное время задержки, если не обнаружили - 30 секунд
                wait_time = delay_seconds if delay_seconds is not None else 30
                print(f"Достигнут лимит. Ждём {wait_time} сек...")
                time.sleep(wait_time)
            else:
                raise e

    return response.text.strip()

# Главная функция
def main():
    configure_api()
    model = genai.GenerativeModel("gemini-1.5-flash")


    # Загружаем текст
    try:
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
    except UnicodeDecodeError:
        with open(TEXT_FILE, "r", encoding="cp1251") as f:
            full_text = f.read()

    # Разбиваем на блоки
    blocks = split_text_blocks(full_text)
    print(f"Найдено блоков: {len(blocks)}")

    dataset = []

    for i, block in enumerate(blocks, 1):
        try:
            output = analyze_block(block, model)
            dataset.append({
                "text_input": block,
                "output": output
            })
            print(f"[{i}] ✓ Обработано")
        except Exception as e:
            print(f"[{i}] ⚠ Ошибка: {e}")

    # Сохраняем в JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    time.sleep(1)  # Дать системе немного времени завершить соединения
    print(f"\n✅ Готово! Сохранено в {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
