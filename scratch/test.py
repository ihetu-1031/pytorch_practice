from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random


base_url = "https://baike.baidu.com"
his = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]

for i in range(20):
    if his:
        url = base_url + his[-1]
    else:
        print("历史链接列表为空，停止执行。")
        break

    try:
        html = urlopen(url).read().decode('utf-8')
        soup = BeautifulSoup(html, features='lxml')
        print(i, soup.find('h1').get_text(), '    url: ', his[-1])

        # find valid urls
        sub_urls = soup.find_all("a", {"target": "_blank", "href": re.compile("/item/(%.{2})+$")})

        if len(sub_urls) != 0:
            his.append(random.sample(sub_urls, 1)[0]['href'])
        else:
            print("没有找到有效的子链接，保持当前链接不变。")
    except Exception as e:
        print(f"错误: {e}")
        break  # 结束循环或处理错误

