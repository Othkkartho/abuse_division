from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

browser = webdriver.Chrome('C:/chromedriver.exe')


def save_list_to_txt(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(str(item) + '|1\n')
    print('save list')


def move_page(page):
    link = f'https://www.ilbe.com{page}'
    return link


file_path = "../../data/link.txt"

with open(file_path, "r", encoding="utf-8") as f:
    view_list = [line.strip() for line in f.readlines()]

total_list = []

for i in view_list:
    url = move_page(i)

    try:
        browser.get(url)
    except TimeoutError:
        break
    try:
        time.sleep(1)
        page_bar = browser.find_elements(By.CSS_SELECTOR, 'div.paginate')[0]
    except IndexError:
        if browser.find_elements(By.CSS_SELECTOR, 'body > div > a > img')[0].accessible_name == '페이지를 찾을 수 없습니다.':
            continue
    pages = page_bar.find_elements(By.CSS_SELECTOR, 'a')
    page_now = page_bar.find_elements(By.CSS_SELECTOR, '.page-on')[0].text

    for j in range(0, (len(pages) - 4)):
        k = j+3
        css = f'#comment_wrap_in > div.paginate > a:nth-child({k})'
        if page_now != '1':
            time.sleep(1)
            browser.find_element(By.CSS_SELECTOR, css).click()
            time.sleep(1)
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        try:
            content = soup.find_all('span', attrs={'class': 'cmt'})
            for texts in content:
                texts = texts.text
                if texts == '삭제 된 댓글입니다.' or texts == '':
                    pass
                else:
                    texts = texts.replace('\n', ' ')
                    total_list.append(texts)
        except IndexError:
            pass

        print("리스트 개수: ", len(total_list))

save_list_to_txt(total_list, '../../data/dataset_libe.txt')

