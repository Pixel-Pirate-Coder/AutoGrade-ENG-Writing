# Оценка уровня Writing по критериям ЕГЭ

 Куратор: Вольф Елена  
 Студенты: Бурлова Альбина, Андрейко Максим, Василенко Павел.

 Облако проекта:  
 https://console.cloud.yandex.ru/folders/b1g58asoc215383g64vr?section=dashboard

 Я.Диск:  
 https://disk.yandex.ru/d/HytFA5aiuGiO_g

 Датасет:  
 https://docs.google.com/spreadsheets/d/1m0mc1H7ULIZ2HEkT4dha_XRmRjt0gWJ8aht_GJ2lxfw/edit#gid=0

 Предлагается создать сервис (web/tg bot), который бы оценивал ответы к заданиям части Writing в ЕГЭ по английскому языку, выставляя баллы по критериям ЕГЭ. Всего 3 задания разного формата, для которых необходимо предоставить письменный ответ в соответствии с критериями задания. 

- Задание 1: «Электронное письмо личного характера».  
Необходимо ответить на вопросы в письме и задать свои адресанту. Всего 3 критерия, выставляется от 0 до 2 баллов по каждому критерию. Максимальный балл — 6.
- Задание 2: «Эссе с элементами рассуждения на основе таблиц/диаграмм».  
Анализ данных по указанной теме и изложение собственных мыслей. Всего 5 критериев, выставляется от 0 до 3 баллов с 1 по 4 критерий, за 5 критерий - от 0 до 2 баллов. Максимальный балл — 14. 

 Каждому заданию соответствуют собственные критерии. Для уменьшения сложности проекта можно не раскладывать ответ по критериям, а выставлять общий балл за задание.
Подробно расписанные критерии и примеры заданий: https://disk.yandex.ru/d/HytFA5aiuGiO_g

 Могут возникнуть сложности со 2 заданием, где условие частично предоставляется в виде изображения (нужно постараться решить данную проблему).
 
 Примерный план работы (в ближайшем будущем задачи дополнятся:
1) Собрать письменные работы и соответствующие им оценки:  
   - Данные в виде сообщений ВК/ТГ, текстовые файлы формата pdf/doc и выставленные оценки с комментариями и без;   
2) Ручной перенос данных в Excel-таблицу:  
   - Возможный парсинг данных из диалога ВК/ТГ или файла;  
4) Обучить модель DL, классифицирующую по баллам;  
5) Создание сайта или телеграм-бота с возможностью загрузки текстового файла или печати в отдельном окне с целью получения итоговой оценки;  
6) Интеграция модели с сервисом.



Организация проекта
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
