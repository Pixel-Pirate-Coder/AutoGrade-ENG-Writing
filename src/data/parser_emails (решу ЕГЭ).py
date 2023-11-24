import pandas as pd
import requests
from bs4 import BeautifulSoup  # не спрашивайте, почему BeautifulSoup
from time import sleep
import random
import re
import openpyxl

URL_start = 'https://en-ege.sdamgia.ru/'
task_list = []
answer_list = []
URL_list = []

for text in range(10956, 11024):
    sleep(random.random())

    URL = URL_start + f'problem?id={text}'
    page = requests.get(URL)
    soup = BeautifulSoup(page.text, 'html.parser')

    curr_task = [task.get_text()
                 for task in soup.find_all('i') if '.' in task.get_text()][0]
    task_list.append(curr_task)

    sleep(random.random())

    curr_answer = [task.get_text()
                   for task in soup.find_all('p') if 'Dear' in task.get_text()][0]
    answer_list.append(curr_answer)

    URL_list.append(URL)

data = {'URL': URL_list, 'Task': task_list, 'Answer': answer_list}
df = pd.DataFrame(data)
df = df.drop_duplicates(subset='Answer')
df.to_excel('C:/Users/SAMAROVEC/Desktop/emails.xlsx', index=False)