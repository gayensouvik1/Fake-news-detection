import requests
from bs4 import BeautifulSoup

URL = 'https://www.google.com/search?pz=1&cf=all&ned=us&hl=en&tbm=nws&gl=us&as_q={query}&as_occt=any&as_drrb=b&as_mindate={month}%2F%{from_day}%2F{year}&as_maxdate={month}%2F{to_day}%2F{year}&tbs=cdr%3A1%2Ccd_min%3A3%2F1%2F13%2Ccd_max%3A3%2F2%2F13&as_nsrc=Gulf%20Times&authuser=0'

def extract_URL(**params):
    response = requests.get(URL.format(**params))
    return  str(response.content)


def run(**params):

    str = extract_URL(query="Demonetisation", month=1, from_day=1, to_day=1, year=17)

    soup = BeautifulSoup(str, 'html.parser')
    for link in soup.find_all('a'):
        mylink = link.get('href')
        print(type(mylink))

run()