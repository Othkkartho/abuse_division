from bs4 import BeautifulSoup
from selenium import webdriver
import time

browser = webdriver.Chrome('C:/chromedriver.exe')


def save_list_to_txt(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(str(item) + '\n')


def move_page(page):
    link = f'https://www.ilbe.com/list/polilbe?page={page}&listStyle=list'
    return link

total_list = []
hrefs = []

for i in range(20, 0, -1):
    url = move_page(i)
    time.sleep(1.5)

    try:
        browser.get(url)
    except TimeoutError:
        break

    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    try:
        link = soup.select('a.subject')

        for j in link:
            hrefs.append(j.attrs['href'])

        for href in hrefs:
            total_list.append(href)

        print("리스트 개수: ", len(total_list))
    except IndexError:
        pass

save_list_to_txt(total_list, '../../data/link.txt')
