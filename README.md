# Оценка уровня Writing по критериям ЕГЭ

 Куратор: Вольф Елена  
 Студенты: Бурлова Альбина, Андрейко Максим, Василенко Павел.

Письмо ЕГЭ по английскому языку — одно из заданий с развернутым ответом письменной части экзамена. Данное задание содержит отрывок письма от друга по переписке. Предлагается написать ответ с соблюдением определенных критериев:

К1 - Решение коммуникативной задачи (даны ответы + заданы вопросы + вежливость и стиль);  
К2 - Организация текста (количество абзацев + логичность);  
К3 - Языковое оформление текста (грамматика + лексика + пунктуация).

По каждому критерию выставляется от 0 до 2 баллов. В общей сумме можно получить максимум 6 баллов.  
Более подробно по критерии: https://disk.yandex.ru/i/3pNnqUuh3D887w

 Создан сервис (API) и подключенный к нему UI (streamlit-приложение и телеграм-бот), которые оценивабт ответы к заданиям части Writing в ЕГЭ по английскому языку, выставляя баллы по критериям ЕГЭ. 

Ссылки:

- [Dataset](https://docs.google.com/spreadsheets/d/1m0mc1H7ULIZ2HEkT4dha_XRmRjt0gWJ8aht_GJ2lxfw/edit#gid=0);
- [Streamlit app](https://app-for-autograde-eng-letter.streamlit.app);
- [Telegram bot](https://t.me/letter_checker_bot);
- [Generated Data - OpenAI API](https://disk.yandex.ru/d/j9CCiZQFpZMTPQ);
- [Fine-tuned BERT](https://disk.yandex.ru/d/5MBlWdXOSiJWuw), [FLAN-T5](https://disk.yandex.ru/d/m8rbGP77RMLoBg);
- [Presentation](https://docs.google.com/presentation/d/1EMiHkaB_kKYvVICD9-9Y8n7eeou9bSIYI5NnJ4Nqhmo/edit?usp=sharing).

# Демонстрация работы сервиса

![]()

![]()

# Архитектура сервиса
 ![Архитектура сервиса](images/service_diagram.png)

Организация проекта
------------

    ├── LICENSE
    ├── README.md                                <- Описание проекта
    ├── data
    │   ├── interim                              <- Intermediate data that has been transformed.
    │   ├── processed                            <- The final, canonical data sets for modeling.
    │   └── raw                                  <- The original, immutable data dump.
    │
    │
    ├── notebooks                                <- Jupyter-ноутбуки. Правила наименования: номер (для учета порядка),
    │                                               инициалы автора, и через `-` краткое название, например
    │                                               `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── production                               <- Git-субмодули с исходным кодом сервисов продуктивизации
    │   ├── autograde_api                        <- API-сервис для обработки запросов
    │   ├── EGE-Writing-Autograde-Bot            <- Телеграм-бот для проверки писем
    │   └── Streamli-for-autograde-eng-letter    <- Web-UI для использования модели, и EDA
    │
    │ 
    ├── src                                      <- Исходный код
    │   ├── __init__.py                          <- Для инициализации папки как модуля
    │   │
    │   └── data                                 <- Код для парсинга, чтения, загрузки данных
    │       ├── parser_email_cloudtext.py        <- Парсер писем с ресурса cloudtext
    │       ├── parser_emails_reshu_ege.py       <- Парсер писем с ресурса РЕШУ ЕГЭ
    │       └──read_raw_data.py                  <- Чтение данных из gsheet
    │
    └── tox.ini                                   <- tox файл настроек для запуска tox; см. tox.readthedocs.io


--------
