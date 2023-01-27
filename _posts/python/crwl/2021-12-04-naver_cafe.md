---
layout : post
title : "[Crawling] 특정한 날짜 간격으로 Naver VIEW (Blog, Cafe) 크롤링하기"
img : python/crwl/naver.png
categories : 
    - [python-crwl]
tag : [Crawling, 크롤링, BeautifulSoup, Python, Naver, 네이버]
toc : true
---

<br/>

## Crawling Code

<br/>

```python
from selenium import webdriver as wd        
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import datetime as dt
import urllib.parse             
import time


def extract_text(posts):            # regex 로 유효한 텍스트만 정제
    tagout = re.compile('<.*?>')
    unicodeout = re.compile(r'"[\\u]%d{4,5}"')
    ps = []
    for p in posts:
        p = re.sub(tagout, "", str(p))
        p = re.sub(unicodeout, "", p)
        p = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", p)
        ps.append(p)
    return ps


def get_posts(url, startdate):
    driver.get(url)
    time.sleep(5)

    html = driver.page_source
    soup = BeautifulSoup(html,'html.parser')
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    weekly_post = []
    weeklyfreq = 0
    posts = []

    tmp = soup.find_all("a", {'class' : "api_txt_lines total_tit _cross_trigger"})
    posts += extract_text(tmp)
    weeklyfreq += len(tmp)

    tmp = soup.find_all("div",  {'class' : "total_group"})
    posts += extract_text(tmp)
    weeklyfreq += len(tmp)

    while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height != last_height:
                html = driver.page_source
                soup = BeautifulSoup(html,'html.parser')
                
                tmp = soup.find_all("a", {'class' : "api_txt_lines total_tit _cross_trigger"})
                posts += extract_text(tmp)
                weeklyfreq += len(tmp)

                tmp = soup.find_all("div",  {'class' : "total_group"})
                posts += extract_text(tmp)
                weeklyfreq += len(tmp)
            else:                                   # 일별로 검색 후 끝까지 scroll 다 내림
                weekly_post.append([startdate, weeklyfreq, posts])
                break

            last_height = new_height
    
    return weekly_post


def createDF(contents, keyword, tag):
    import pandas as pd
    df = pd.DataFrame(contents, columns=['Date', 'Weekly Frequency', 'Tweets'])
    df.to_excel("NAVER_" + keyword + "_" + str(tag) + ".xlsx")


if __name__ == "__main__":

    driver = wd.Chrome("chromedriver.exe")
    keywords = ["코로나, 감정", "코로나, 기분", "코로나, 일상"]      # 크롤링 키워드
    years = [2020, 2021]        # 크롤링하고자 하는 년도

    for keyword in keywords:
            for year in years:
                contents = []
                if year == 2020: limit = 13
                elif year == 2021: limit = 11
                for m in range(1, limit):
                    i = m
                    if len(str(m)) == 1: m = str(0) + str(m)      # 20210607 이런 형식으로 날짜 입력해줘야 함
                    base = str(year) + str(m)
                    startdate = base + "01"
                    
                    for d in ["07", "14", "21", "28"]:      # 7일 간격으로
                        middate = base + d
                        url = f"https://search.naver.com/search.naver?where=view&query={keyword}&sm=tab_opt&nso=so%3Ar%2Cp%3Afrom{startdate}to{middate}%2Ca%3Aall&mode=normal&main_q=&st_coll=&topic_r_cat="
                        contents += get_posts(url, startdate)
                        startdate = middate
                        # print(i)
                        
                    if i == 6 or i == limit-1:
                        if i == 6: add = "_상반기"
                        elif i == limit-1: add = "_하반기"
                        tag = str(year) + "년" + add
                        createDF(contents, keyword, tag)
                        contents = []
```

<br/>

- html class 가 꽤나 자주 변경되는 거 같으니 find_all 을 사용할 때 원하는 요소의 html class 를 잘 확인해봐야 한다.
- 검색 결과로 나온 게시글들 전문을 가져오려면 시간이 너무 오래 걸릴 거 같아 포스팅 창을 열지않고 제목, 미리보기로 노출되는 부분만 가져왔다. 