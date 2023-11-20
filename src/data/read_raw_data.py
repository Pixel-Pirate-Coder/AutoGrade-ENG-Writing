import gspread as gs
import pandas as pd
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def read_from_gsheet():
    gc = gs.service_account(filename=os.environ['PATH_TO_GOOGLE_SHEET_CREDENTIALS'])
    sheet = gc.open_by_url(os.environ['DATABASE_URL'])
    ws = sheet.worksheet('Dataset')
    return pd.DataFrame(ws.get_all_records())

def drop_unwanted_data(df: pd.DataFrame):
    columns_to_drop = ['Comments', 'origin', 'task_image', 'score_image', 'Overall_score']
    columns = ['Solving a communicative task', 'Text structure', 'Use of English (for emails)', 'Lexis (essay)',
               'Grammatical accuracy (essay)', 'Punctuation and spelling (essay)']

    # Удаляем лишние столбцы (на данный момент)
    df = df.drop(columns=columns_to_drop)

    # Удаляем строки баллов, если там сожержатся другие значения, кроме int и float (избавимся от string)
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

def split_types(df: pd.DataFrame):
    # Следующий этап - разделим на письма и эссе
    essay_df = df[data[' Type'] == 'Essay']
    email_df = df[data[' Type'] == 'Email']

    # и удалим лишние колонки для каждого типа
    essay_df = essay_df.drop(columns='Use of English (for emails)')
    email_df = email_df.drop(
        columns=['Lexis (essay)', 'Grammatical accuracy (essay)', 'Punctuation and spelling (essay)'])

    # Убираем NaN
    essay_df = essay_df.dropna()
    email_df = email_df.dropna()

    return essay_df, email_df

raw_data = read_from_gsheet()
data = drop_unwanted_data(raw_data)
_, email_data = split_types(data)
print(email_data.info())



