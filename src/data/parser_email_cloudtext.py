from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time
import os
import pandas as pd

email = os.getenv("LOGIN")
password = os.getenv("PASS")
email_col_names = ('Solving a communicative task', 'Text structure', 'Use of English (for emails)')
data = pd.DataFrame()


def get_next_btn_webelem(driver):
    return driver.find_element(By.XPATH, "//a[@class='page-link' and text()='Вперед']")


def get_next_btn_class(next_btn):
    return next_btn.find_element(By.XPATH, './..').get_attribute('class')


driver = webdriver.Chrome()
driver.maximize_window()

driver.get('https://nezagorami-eng.cloudtext.ru/tasks/35/stats')
time.sleep(2)
driver.find_element(By.XPATH, "//input[@type='email']").send_keys(email)
driver.find_element(By.CLASS_NAME, 'btn').click()
time.sleep(1)
driver.find_element(By.XPATH, "//input[@type='password']").send_keys(password)
driver.find_element(By.CLASS_NAME, 'btn').click()
time.sleep(5)

next_btn_class = get_next_btn_class(get_next_btn_webelem(driver))
while next_btn_class != 'page-item next disabled':
    time.sleep(1)
    all_rows_in_page = driver.find_elements(By.XPATH, "//*[@id='app']/main/div[2]/div[2]/div[1]/div[2]/div/table/tbody/tr")
    assessed_links = []
    # assessed_links = [row.find_element(By.TAG_NAME, 'a').get_attribute('href') for row in all_rows_in_page if row.find_element(By.TAG_NAME, 'span').text == 'Проверена']
    for row in all_rows_in_page:
        if row.find_element(By.TAG_NAME, 'span').text == 'Проверена':
            assessed_links.append(row.find_element(By.TAG_NAME, 'a').get_attribute('href'))
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    for link in assessed_links:
        curr_data = {'Link': link}
        print(link)
        driver.get(link)
        time.sleep(2)
        task = driver.find_element(By.XPATH, "//div[@class='work-field-question']")
        answer = driver.find_element(By.XPATH, "//div[@class='work-field-text']")
        answer_paragraphs = answer.find_elements(By.TAG_NAME, "p")
        clean_answer = ''
        for par in answer_paragraphs:
            try:
                bold_text = list(map(lambda x: x.text, par.find_elements(By.TAG_NAME, "b")))
            except:
                bold_text = []
            try:
                italic_text = list(map(lambda x: x.text, par.find_elements(By.TAG_NAME, "i")))
            except:
                italic_text = []
            par_text = par.text
            for bold_fragment in bold_text:
                par_text = par_text.replace(bold_fragment, ' ')
            for ital_fragment in italic_text:
                par_text = par_text.replace(ital_fragment, ' ')

            clean_answer += '\n\n' + par_text
        curr_data['Text'] = clean_answer
        time.sleep(1)
        all_selections = driver.find_elements(By.TAG_NAME, 'select')
        all_strongs = driver.find_elements(By.XPATH, "//div[@class='value-ball']/strong")

        if all_selections:
            selected_scores = list(map(lambda x: int(Select(x).all_selected_options[0].text.split('из')[0]), all_selections))
            scores = dict(zip(email_col_names, selected_scores))
        elif all_strongs:
            selected_scores = list(map(lambda x: int(x.text.split('из')[0]), all_strongs))
            scores = dict(zip(email_col_names, selected_scores))
        else:
            scores = dict(zip(email_col_names, list()))
        curr_data.update(scores)

        data = pd.concat([data, pd.DataFrame(curr_data, index=[0])], axis=0)
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    next_btn = get_next_btn_webelem(driver)
    next_btn_class = get_next_btn_class(next_btn)
    if next_btn_class == 'page-item next':
        next_btn.click()


driver.quit()
data.to_excel('email_household_chores.xlsx')
