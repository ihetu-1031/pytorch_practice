from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random


url = "https://en.wikipedia.org/wiki/Web_scraping"

html = urlopen(url).read().decode('utf-8')
soup = BeautifulSoup(html, features='lxml')

sub_urls = soup.find_all("a", href = re.compile("^(/wiki/)((?!:).)*$"))
for urls in sub_urls:
    if not re.search("\.(jpg|JPG)$", urls["href"]):
        print (urls.get_text(),"------>",urls["href"])