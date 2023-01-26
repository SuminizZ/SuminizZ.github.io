---
title : "[Data] Sentiment Analysis with BiLSTM Model trained with Multiple Datasets"
categories : 
    - Bonfire
tag : [Bonfire, 코로나, BiLSTM, NLP, 감정분석]
toc : true
---

## **필요한 Module Import**

```python
!pip install konlpy
```

    Collecting konlpy
      Downloading konlpy-0.5.2-py2.py3-none-any.whl (19.4 MB)
    [K     |████████████████████████████████| 19.4 MB 1.4 MB/s 
    [?25hRequirement already satisfied: tweepy>=3.7.0 in /usr/local/lib/python3.7/dist-packages (from konlpy) (3.10.0)
    Collecting JPype1>=0.7.0
      Downloading JPype1-1.3.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (448 kB)
    [K     |████████████████████████████████| 448 kB 42.7 MB/s 
    [?25hRequirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.7/dist-packages (from konlpy) (4.2.6)
    Collecting beautifulsoup4==4.6.0
      Downloading beautifulsoup4-4.6.0-py3-none-any.whl (86 kB)
    [K     |████████████████████████████████| 86 kB 5.4 MB/s 
    [?25hRequirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.7/dist-packages (from konlpy) (1.19.5)
    Collecting colorama
      Downloading colorama-0.4.4-py2.py3-none-any.whl (16 kB)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from JPype1>=0.7.0->konlpy) (3.10.0.2)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tweepy>=3.7.0->konlpy) (1.15.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tweepy>=3.7.0->konlpy) (1.3.0)
    Requirement already satisfied: requests[socks]>=2.11.1 in /usr/local/lib/python3.7/dist-packages (from tweepy>=3.7.0->konlpy) (2.23.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy) (3.1.1)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2021.10.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.24.3)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.7/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.7.1)
    Installing collected packages: JPype1, colorama, beautifulsoup4, konlpy
      Attempting uninstall: beautifulsoup4
        Found existing installation: beautifulsoup4 4.6.3
        Uninstalling beautifulsoup4-4.6.3:
          Successfully uninstalled beautifulsoup4-4.6.3
    Successfully installed JPype1-1.3.0 beautifulsoup4-4.6.0 colorama-0.4.4 konlpy-0.5.2
    


```python
import re
import urllib.request
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")

from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

## **DATASET 정리**
- 감정분석 모델을 학습시키기 위한 감정 라벨링 데이터셋
- 네이버 영화 리뷰 데이터셋, 네이버 쇼핑몰 리뷰 데이터셋, 스팀 게임 리뷰 총 3개의 데이터셋을 활용했다.

### **1. Naver Movie Review Datset**


```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",\
                           filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",\
                           filename="ratings_test.txt")
```




    ('ratings_test.txt', <http.client.HTTPMessage at 0x7f41000f26d0>)




```python
movie_test_data = pd.read_table("/content/ratings_test.txt")
movie_train_data = pd.read_table("/content/ratings_train.txt")
```


```python
movie_train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>document</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9976970</td>
      <td>아 더빙.. 진짜 짜증나네요 목소리</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3819312</td>
      <td>흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10265843</td>
      <td>너무재밓었다그래서보는것을추천한다</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9045019</td>
      <td>교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6483659</td>
      <td>사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>149995</th>
      <td>6222902</td>
      <td>인간이 문제지.. 소는 뭔죄인가..</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149996</th>
      <td>8549745</td>
      <td>평점이 너무 낮아서...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>149997</th>
      <td>9311800</td>
      <td>이게 뭐요? 한국인은 거들먹거리고 필리핀 혼혈은 착하다?</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149998</th>
      <td>2376369</td>
      <td>청춘 영화의 최고봉.방황과 우울했던 날들의 자화상</td>
      <td>1</td>
    </tr>
    <tr>
      <th>149999</th>
      <td>9619869</td>
      <td>한국 영화 최초로 수간하는 내용이 담긴 영화</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>150000 rows × 3 columns</p>
</div>



### **2. Naver ShoppingMall Review Dataset**



```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", \
                    filename="shop_ratings.txt")
```




    ('shop_ratings.txt', <http.client.HTTPMessage at 0x7f410008d410>)




```python
shop_data = pd.read_table("/content/shop_ratings.txt")
```


```python
shop_data = shop_data.rename(columns={'5' : "label", "배공빠르고 굿" : "document"})
shop_data['label'] = shop_data['label'].apply(lambda x : 1 if x > 3 else 0)
```


```python
shop_data.shape
```




    (199999, 2)




```python
shop_train_data = shop_data.iloc[:150000, :]
shop_test_data = shop_data.iloc[150000: :]
```


```python
shop_train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 2 columns):
     #   Column    Non-Null Count   Dtype 
    ---  ------    --------------   ----- 
     0   label     150000 non-null  int64 
     1   document  150000 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 2.3+ MB
    


```python
shop_test_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 49999 entries, 150000 to 199998
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   label     49999 non-null  int64 
     1   document  49999 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 781.4+ KB
    


```python
shop_train_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>document</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>택배가 엉망이네용 저희집 밑에층에 말도없이 놔두고가고</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>아주좋아요 바지 정말 좋아서2개 더 구매했어요 이가격에 대박입니다. 바느질이 조금 ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>선물용으로 빨리 받아서 전달했어야 하는 상품이었는데 머그컵만 와서 당황했습니다. 전...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>민트색상 예뻐요. 옆 손잡이는 거는 용도로도 사용되네요 ㅎㅎ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>비추합니다 계란 뒤집을 때 완전 불편해요 ㅠㅠ 코팅도 묻어나고 보기엔 예쁘고 실용적...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>주문을 11월6에 시켰는데 11월16일에 배송이 왔네요 ㅎㅎㅎ 여기 회사측과는 전화...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>넉넉한 길이로 주문했는데도 안 맞네요 별로예요</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>보폴이 계속 때처럼 나오다가 지금은 안나네요~</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>110인데 전문속옷브랜드 위생팬티105보다 작은듯해요. 불편해요. 밴딩부분이 다 신...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>사이즈도 딱이고 귀엽고 넘 좋아요 ㅎㅎ</td>
    </tr>
  </tbody>
</table>
</div>




```python
shop_train_data['label'].value_counts()
```




    0    75109
    1    74891
    Name: label, dtype: int64




```python
shop_test_data['label'].value_counts()
```




    1    25071
    0    24928
    Name: label, dtype: int64



### **3. Game Review Dataset**


```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", \
                           filename="game_ratings.txt")
```




    ('game_ratings.txt', <http.client.HTTPMessage at 0x7f41000aa3d0>)




```python
game_data = pd.read_table("/content/game_ratings.txt")
```


```python
game_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>노래가 너무 적음</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>돌겠네 진짜. 황숙아, 어크 공장 그만 돌려라. 죽는다.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>막노동 체험판 막노동 하는사람인데 장비를 내가 사야돼 뭐지</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>차악!차악!!차악!!! 정말 이래서 왕국을 되찾을 수 있는거야??</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>시간 때우기에 좋음.. 도전과제는 50시간이면 다 깰 수 있어요</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>역시 재미있네요 전작에서 할수 없었던 자유로운 덱 빌딩도 좋네요^^</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99994</th>
      <td>0</td>
      <td>한글화해주면 10개산다</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>0</td>
      <td>개쌉노잼 ㅋㅋ</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>0</td>
      <td>노잼이네요... 30분하고 지웠어요...</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>1</td>
      <td>야생을 사랑하는 사람들을 위한 짧지만 여운이 남는 이야기. 영어는 그리 어렵지 않습니다.</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>1</td>
      <td>한국의 메탈레이지를 떠오르게한다 진짜 손맛으로 하는게임</td>
    </tr>
  </tbody>
</table>
<p>99999 rows × 2 columns</p>
</div>




```python
game_data = game_data.rename(columns={'0' : "label", '노래가 너무 적음' : "document"})
```


```python
game_train_data = game_data.iloc[:80000, :]
game_test_data = game_data.iloc[80000: :]
```


```python
game_train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 80000 entries, 0 to 79999
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   label     80000 non-null  int64 
     1   document  80000 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 1.2+ MB
    


```python
game_test_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19999 entries, 80000 to 99998
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   label     19999 non-null  int64 
     1   document  19999 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 312.6+ KB
    


```python
game_train_data['label'].value_counts()
game_test_data['label'].value_counts()
```




    1    10003
    0     9996
    Name: label, dtype: int64



### **모든 데이터셋 preprocess 및 merge**


```python
def preprocess_df(train_data, test_data):

    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data.dropna(inplace=True)
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train_data['document'] = train_data['document'].str.replace('^ +', "")   
    train_data['document'].replace("", np.nan, inplace=True)      
    train_data.dropna(inplace=True)
    train_data.drop_duplicates(subset=['document'], inplace=True)

    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data.dropna(inplace=True)
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    test_data['document'] = test_data['document'].str.replace('^ +', "")   
    test_data['document'].replace("", np.nan, inplace=True)      
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(subset=['document'], inplace=True)

    return train_data, test_data
```


```python
movie_train_data, movie_test_data = preprocess_df(movie_train_data, movie_test_data)
shop_train_data, shop_test_data = preprocess_df(shop_train_data, shop_test_data)
game_train_data, game_test_data = preprocess_df(game_train_data, game_test_data)
```


```python
df_list = [movie_train_data, movie_test_data, shop_train_data, shop_test_data, game_train_data, game_test_data]

for df in df_list:
    print(df.shape)
```

    (143620, 3)
    (48389, 3)
    (149638, 2)
    (49936, 2)
    (79517, 2)
    (19970, 2)
    


```python
# column 순서 바꾸고 id column 삭제

movie_train_data.drop('id', axis=1, inplace=True)
movie_train_data = movie_train_data[['label', 'document']]
movie_test_data.drop('id', axis=1, inplace=True)
movie_test_data = movie_test_data[['label', 'document']]
```


```python
movie_train_data
movie_test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>document</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>굳 ㅋ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>뭐야 이 평점들은 나쁘진 않지만 점 짜리는 더더욱 아니잖아</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>지루하지는 않은데 완전 막장임 돈주고 보기에는</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>만 아니었어도 별 다섯 개 줬을텐데 왜 로 나와서 제 심기를 불편하게 하죠</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>음악이 주가 된 최고의 음악영화</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>1</td>
      <td>오랜만에 평점 로긴했네ㅋㅋ 킹왕짱 쌈뽕한 영화를 만났습니다 강렬하게 육쾌함</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>0</td>
      <td>의지 박약들이나 하는거다 탈영은 일단 주인공 김대희 닮았고 이등병 찐따</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>0</td>
      <td>그림도 좋고 완성도도 높았지만 보는 내내 불안하게 만든다</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>0</td>
      <td>절대 봐서는 안 될 영화 재미도 없고 기분만 잡치고 한 세트장에서 다 해먹네</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>0</td>
      <td>마무리는 또 왜이래</td>
    </tr>
  </tbody>
</table>
<p>48389 rows × 2 columns</p>
</div>




```python
# movie_train_data
# shop_train_data
game_train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>document</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>돌겠네 진짜 황숙아 어크 공장 그만 돌려라 죽는다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>막노동 체험판 막노동 하는사람인데 장비를 내가 사야돼 뭐지</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>차악차악차악 정말 이래서 왕국을 되찾을 수 있는거야</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>시간 때우기에 좋음 도전과제는 시간이면 다 깰 수 있어요</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>역시 재미있네요 전작에서 할수 없었던 자유로운 덱 빌딩도 좋네요</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79995</th>
      <td>0</td>
      <td>어느정도 재미가 있긴 하지만 지금 하기엔 너무 짜증나는 점이 많은 게임 돈모아서 다...</td>
    </tr>
    <tr>
      <th>79996</th>
      <td>1</td>
      <td>평가 읽을 시간에 듀얼하셈 듀얼근 손실 옴 반박시 듀얼</td>
    </tr>
    <tr>
      <th>79997</th>
      <td>1</td>
      <td>이후로는 게임이 상당히 변화했고 십자군 시스템과 혈통 시스템은 상당히 맘에 듭니다 ...</td>
    </tr>
    <tr>
      <th>79998</th>
      <td>1</td>
      <td>냥겜</td>
    </tr>
    <tr>
      <th>79999</th>
      <td>1</td>
      <td>달껄룩의 찬란한 모험기</td>
    </tr>
  </tbody>
</table>
<p>79517 rows × 2 columns</p>
</div>




```python
# concat all three datasets

train_data = pd.concat([movie_train_data, shop_train_data, game_train_data], join='outer')
test_data = pd.concat([movie_test_data, shop_test_data, game_test_data], join='outer')
```


```python
train_data.shape
# test_data.shape
```




    (372775, 2)




```python
train_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>document</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>아 더빙 진짜 짜증나네요 목소리</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>너무재밓었다그래서보는것을추천한다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>교도소 이야기구먼 솔직히 재미는 없다평점 조정</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>사이몬페그의 익살스런 연기가 돋보였던 영화스파이더맨에서 늙어보이기만 했던 커스틴 던...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>막 걸음마 뗀 세부터 초등학교 학년생인 살용영화ㅋㅋㅋ별반개도 아까움</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>원작의 긴장감을 제대로 살려내지못했다</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>별 반개도 아깝다 욕나온다 이응경 길용우 연기생활이몇년인지정말 발로해도 그것보단 낫...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>액션이 없는데도 재미 있는 몇안되는 영화</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>왜케 평점이 낮은건데 꽤 볼만한데 헐리우드식 화려함에만 너무 길들여져 있나</td>
    </tr>
  </tbody>
</table>
</div>



---

## **Tokenization & Padding**


```python
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
stopwords += ['이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같', '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하']
stopwords = list(set(stopwords))
# stopwords
```


```python
from konlpy.tag import Okt      
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

okt = Okt()
```


```python
x_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sent = okt.morphs(sentence, stem=True)      # 형태소 별로 문장 분할
    tokenized_sent = [word for word in tokenized_sent if word not in stopwords]   # 불용어 제거
    x_train.append(tokenized_sent)
```

    100%|██████████| 372775/372775 [24:49<00:00, 250.33it/s]
    


```python
x_train[:2]
```




    [['아', '더빙', '진짜', '짜증나다', '목소리'],
     ['흠', '포스터', '보고', '초딩', '영화', '줄', '오버', '연기', '조차', '가볍다', '않다']]




```python
x_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sent = okt.morphs(sentence, stem=True)      # 형태소 별로 문장 분할
    tokenized_sent = [word for word in tokenized_sent if word not in stopwords]   # 불용어 제거
    x_test.append(tokenized_sent)
```

    100%|██████████| 118295/118295 [09:59<00:00, 197.30it/s]
    


```python
test_data['document'][:2]
x_test[:2]
```




    [['굳다', 'ㅋ'], ['뭐', '야', '평점', '나쁘다', '않다', '점', '짜다', '리', '더', '더욱', '아니다']]




```python
x_train_raw = x_train
x_test_raw = x_test
```

### **Tokenizer 로 단어사전 만들기 & 정수 Encoding**
- 단어사전은 train_data 로 만들어야 함


```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)  

tokenizer.word_index      # 빈도수가 높은 단어별로 작은 수의 양의 정수를 부여함
len(tokenizer.word_index)
```




    75147




```python
threshold = 4
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0      
total_freq = 0     
rare_freq = 0       

for key, value in tokenizer.word_counts.items():
    total_freq += value

    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```

    단어 집합(vocabulary)의 크기 : 75147
    등장 빈도가 3번 이하인 희귀 단어의 수: 48485
    단어 집합에서 희귀 단어의 비율: 64.52020706082745
    전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 1.5172733686959021
    


```python
vocab_size = total_cnt - rare_cnt + 1
vocab_size
```




    26663




```python
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(x_train)   # 0번 단어 ~ 19395번 단어까지만 사용
```


```python
# Referring to word_index, text --> numeric sequences  

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
```


```python
print(x_train[:3])
print(x_test[:3])
```

    [[64, 1007, 30, 364, 1329], [897, 1127, 69, 1146, 4, 236, 1843, 99, 1259, 328, 15], [354, 4758, 3973, 4417, 1, 83, 12]]
    [[668, 160], [86, 246, 101, 375, 15, 53, 254, 770, 34, 1074, 20], [151, 15, 117, 861, 140, 102, 68, 315, 176]]
    


```python
len(x_test)
# len(x_test)
```




    118295




```python
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
```


```python
# Remove empty sequence in x_train, which means those seq are composed of only the words that are under set threshold

drop_target = [idx for idx, sentence in enumerate(x_train) if len(sentence) < 1]
x_train = np.delete(x_train, drop_target, axis=0)       # np.delete(array, delete_target_idx, axis=0(row) or 1(col))
y_train = np.delete(y_train, drop_target, axis=0)
```


```python
print(len(x_train))
print(len(y_train))       # 143620 --> 143376
```

    372193
    372193
    

### **Padding**
- 모델이 처리할 수 있도록 X_train과 X_test의 모든 샘플의 길이를 특정 길이로 동일하게 맞춤
- padding_len 


```python
print('리뷰의 최대 길이 :', max(map(len, x_train)))
print('리뷰의 평균 길이 :',sum(map(len, x_train))/len(x_train))
plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```

    리뷰의 최대 길이 : 68
    리뷰의 평균 길이 : 11.614372113392783
    


    
<img src="https://user-images.githubusercontent.com/92680829/144726149-7a4b8670-2bc3-4a47-83e3-e505aafb37ce.png" >
    



```python
def find_padding_len(set_len):
    cnt = 0
    for l in list(map(len, x_train)):
        if l <= set_len:
            cnt += 1
    return (cnt/len(x_train))*100

for set_len in range(10, 50, 5):
    tmp = find_padding_len(set_len)
    print("길이 {0} : {1}%".format(set_len, tmp))
```

    길이 10 : 59.378870639694995%
    길이 15 : 75.60620430798%
    길이 20 : 84.42609076473765%
    길이 25 : 90.24833889944196%
    길이 30 : 94.36824443232409%
    길이 35 : 97.26539725357543%
    길이 40 : 99.18187606967352%
    길이 45 : 99.86969126232896%
    


```python
padding_len = 35
```


```python
x_train = pad_sequences(x_train, maxlen = padding_len, truncating='post', padding='post')
x_test = pad_sequences(x_test, maxlen = padding_len, truncating='post', padding='post')
```


```python
x_train
```




    array([[   64,  1007,    30, ...,     0,     0,     0],
           [  897,  1127,    69, ...,     0,     0,     0],
           [  354,  4758,  3973, ...,     0,     0,     0],
           ...,
           [  470,   824,    51, ...,     0,     0,     0],
           [  688,  1106,     8, ...,     0,     0,     0],
           [19611,  5647,  2779, ...,     0,     0,     0]], dtype=int32)




```python
x_train.shape       # padding 완료
```




    (372193, 35)



---

## **LSTM Training with DATASET**

### **Import Modules**


```python
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```


```python
emb_dim = 100
cur_dim = 26663     # vocab size
padding_len = 35
```

### **Modeling**


```python
model = Sequential([
    Embedding(cur_dim, emb_dim, input_length=padding_len),      
    Bidirectional(LSTM(128, return_sequences=True)),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     (None, 35, 100)           2666300   
                                                                     
     bidirectional_1 (Bidirectio  (None, 35, 256)          234496    
     nal)                                                            
                                                                     
     lstm_3 (LSTM)               (None, 32)                36992     
                                                                     
     dense_1 (Dense)             (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 2,937,821
    Trainable params: 2,937,821
    Non-trainable params: 0
    _________________________________________________________________
    


```python
import pydot
import graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```

<img src="https://user-images.githubusercontent.com/92680829/144725961-81ab2635-f689-4fa6-8d6e-edef15d84330.png" >


```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
```


```python
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss',
                             verbose=1)
```


```python
earlystop = EarlyStopping(monitor="val_loss", 
                          min_delta=0.001,        
                          patience=2)
```

### **Training**


```python
epochs=10
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, earlystop],
                    epochs=epochs)
```

    Epoch 1/10
    11631/11632 [============================>.] - ETA: 0s - loss: 0.3492 - acc: 0.8493
    Epoch 00001: val_loss improved from inf to 0.33655, saving model to my_checkpoint.ckpt
    11632/11632 [==============================] - 428s 37ms/step - loss: 0.3492 - acc: 0.8493 - val_loss: 0.3366 - val_acc: 0.8570
    Epoch 2/10
    11632/11632 [==============================] - ETA: 0s - loss: 0.2986 - acc: 0.8740
    Epoch 00002: val_loss improved from 0.33655 to 0.32506, saving model to my_checkpoint.ckpt
    11632/11632 [==============================] - 429s 37ms/step - loss: 0.2986 - acc: 0.8740 - val_loss: 0.3251 - val_acc: 0.8614
    Epoch 3/10
    11632/11632 [==============================] - ETA: 0s - loss: 0.2634 - acc: 0.8908
    Epoch 00003: val_loss did not improve from 0.32506
    11632/11632 [==============================] - 429s 37ms/step - loss: 0.2634 - acc: 0.8908 - val_loss: 0.3457 - val_acc: 0.8606
    


```python
model.load_weights(checkpoint_path)
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fe9f7682b10>




```python
loss, acc = model.evaluate(x_test, y_test)
```

    3697/3697 [==============================] - 57s 13ms/step - loss: 0.3251 - acc: 0.8614
    


```python
model.save_weights('lstm_model.h5')
```


```python
model.save("model")
```

    WARNING:absl:Found untraced functions such as lstm_cell_3_layer_call_fn, lstm_cell_3_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn while saving (showing 5 of 15). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: model/assets
    

    INFO:tensorflow:Assets written to: model/assets
    WARNING:absl:<keras.layers.recurrent.LSTMCell object at 0x7f4102637a50> has the same name 'LSTMCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.LSTMCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
    WARNING:absl:<keras.layers.recurrent.LSTMCell object at 0x7f4102700d90> has the same name 'LSTMCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.LSTMCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
    WARNING:absl:<keras.layers.recurrent.LSTMCell object at 0x7f41027710d0> has the same name 'LSTMCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.LSTMCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
    

### **모델 성능**

```python
print("Model Loss : {0}\nModel Accuracy : {1}".format(np.round(loss, 4), np.round(acc, 4)))
```

    Model Loss : 0.3251
    Model Accuracy : 0.8614
    

---

## **학습된 모델로 Naver, Twitter 에서 크롤링한 게시글 감정분석 후 엑셀파일 생성**

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
```


```python
naver_emo = pd.read_excel("/content/NAVER_코로나, 감정.xlsx")
naver_mood = pd.read_excel("/content/NAVER_코로나, 기분.xlsx")
naver_daily = pd.read_excel("/content/NAVER_코로나, 일상.xlsx")

twitter_emo = pd.read_excel("/content/Tweets_코로나, 감정.xlsx")
twitter_mood = pd.read_excel("/content/Tweets_코로나, 기분.xlsx")
twitter_daily = pd.read_excel("/content/Tweets_코로나, 일상.xlsx")
```


```python
naver_emo.rename(columns={'Tweets' : 'Contents'}, inplace=True)
naver_mood.rename(columns={'Tweets' : 'Contents'}, inplace=True)
naver_daily.rename(columns={'Tweets' : 'Contents'}, inplace=True)
twitter_emo.rename(columns={'Tweets' : 'Contents'}, inplace=True)
twitter_mood.rename(columns={'Tweets' : 'Contents'}, inplace=True)
twitter_daily.rename(columns={'Tweets' : 'Contents'}, inplace=True)
```


```python
naver_emo['Positive'] = 0
naver_emo['Negative'] = 0

naver_mood['Positive'] = 0
naver_mood['Negative'] = 0

naver_daily['Positive'] = 0
naver_daily['Negative'] = 0

twitter_emo['Positive'] = 0
twitter_emo['Negative'] = 0

twitter_mood['Positive'] = 0
twitter_mood['Negative'] = 0

twitter_daily['Positive'] = 0
twitter_daily['Negative'] = 0
```


```python
naver_emo
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Weekly Frequency</th>
      <th>Contents</th>
      <th>Positive</th>
      <th>Negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20200101</td>
      <td>146</td>
      <td>['년 월 일 비대면온라인개강성경적감정코칭총신대학교평생교육원 전문교육아카데미', '...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20200107</td>
      <td>166</td>
      <td>['스스로 변화를 이끌어내는 셀프코칭 일 감정일기 프로젝트 기 모집 일정 연기', ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20200114</td>
      <td>164</td>
      <td>['왜 나는 항상 먹고나서 후회할까 원데이클래쓰 차 오픈 코로나 사태로 잠정 보류'...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20200121</td>
      <td>890</td>
      <td>[' 우한 발생 신종 코로나 바이러스 폐렴 질환 확산과 주식 시장', '신종코로나바...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20200201</td>
      <td>896</td>
      <td>[' 감정일기  코로나 바이러스', ' 미국 증시의 이격 조정 그리고 신종 코로나 ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>20210921</td>
      <td>896</td>
      <td>[' 위드코로나 길어질수록 내 감정을 해치는 영향들', '생장일지 감정을 기록하는 ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>20211001</td>
      <td>880</td>
      <td>['앵콜전시 화무십일홍 전시  코로나 블루시대 인 작가의 감정과 사유 아트포스터 전...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>20211007</td>
      <td>898</td>
      <td>['앵콜전시 화무십일홍 전시  코로나 블루시대 인 작가의 감정과 사유 아트포스터 전...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>20211014</td>
      <td>894</td>
      <td>['일본인  중국 싫다코로나 속  국민감정 역대 최악', '일본인  중국 싫다코로나...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>20211021</td>
      <td>894</td>
      <td>['일본인  중국 싫다코로나 속  국민감정 역대 최악', '일본인  중국 싫다코로나...</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>88 rows × 5 columns</p>
</div>




```python
def get_sentiment(sen):
    sen = okt.morphs(sen, stem=True)
    sen = [word for word in sen if word not in stopwords]
    encoded = tokenizer.texts_to_sequences([sen])       # label encoding
    padded = pad_sequences(encoded, maxlen=padding_len)       # padding
    # print(padded)
    score = float(model.predict(padded))
    if score >= 0.65:
        return 1
    else:
        return -1
```


```python
pos_sen = []
neg_sen = []
```


```python
def measure_sentiment_for_each_df(df):
    for i, contents in tqdm(enumerate(df['Contents'])):
        content_list = contents.split(',')

        for sen in content_list:
            sen = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", sen)
            tmp_res = get_sentiment(sen)
            if tmp_res == 1:
                df['Positive'][i] += 1
                pos_sen.append(sen)
            else:
                df['Negative'][i] += 1
                neg_sen.append(sen)
```


```python
pd.options.mode.chained_assignment = None
```


```python
df_list = [twitter_mood, twitter_daily]
df_name_list = ["twitter_mood", "twitter_daily"]
# f"{df_name_list[3]}"
```


```python
i = 0
for df in tqdm(df_list):
    measure_sentiment_for_each_df(df)
    df.to_excel(f"{df_name_list[i]}.xlsx")
    keywords = {'pos' : [pos_sen], 'neg': [neg_sen]}
    keywords_df = pd.DataFrame(keywords)
    keywords_df.to_excel("categorized_keywords.xlsx")
    i += 1
```

      0%|          | 0/2 [00:00<?, ?it/s]
    0it [00:00, ?it/s][A
    1it [00:01,  1.30s/it][A
    3it [00:01,  2.60it/s][A
    4it [00:03,  1.16it/s][A
    5it [00:12,  3.78s/it][A
    6it [00:21,  5.31s/it][A
    7it [00:33,  7.36s/it][A
    8it [01:01, 13.89s/it][A
    9it [01:31, 18.66s/it][A
    10it [01:59, 21.54s/it][A
    11it [02:26, 23.26s/it][A
    12it [02:53, 24.31s/it][A
    13it [03:14, 23.56s/it][A
    14it [03:36, 23.08s/it][A
    15it [03:54, 21.41s/it][A
    16it [04:11, 20.03s/it][A
    17it [04:21, 17.03s/it][A
    18it [04:39, 17.24s/it][A
    19it [04:52, 16.00s/it][A
    20it [05:08, 16.17s/it][A
    21it [05:17, 14.08s/it][A
    22it [05:29, 13.37s/it][A
    23it [05:38, 12.14s/it][A
    24it [05:47, 11.10s/it][A
    25it [05:55, 10.01s/it][A
    26it [06:02,  9.13s/it][A
    27it [06:08,  8.37s/it][A
    28it [06:14,  7.62s/it][A
    29it [06:19,  6.78s/it][A
    30it [06:25,  6.57s/it][A
    31it [06:48, 11.43s/it][A
    32it [07:15, 16.07s/it][A
    33it [07:31, 16.04s/it][A
    34it [07:46, 15.73s/it][A
    35it [07:57, 14.47s/it][A
    36it [08:08, 13.33s/it][A
    37it [08:15, 11.53s/it][A
    38it [08:24, 10.64s/it][A
    39it [08:30,  9.43s/it][A
    40it [08:37,  8.52s/it][A
    41it [08:42,  7.54s/it][A
    42it [08:46,  6.46s/it][A
    43it [08:53,  6.65s/it][A
    44it [09:04,  7.94s/it][A
    45it [09:17,  9.46s/it][A
    46it [09:32, 11.26s/it][A
    47it [09:47, 12.12s/it][A
    48it [10:01, 12.81s/it][A
    49it [10:10, 11.70s/it][A
    50it [10:18, 10.69s/it][A
    51it [10:26,  9.88s/it][A
    52it [10:35,  9.49s/it][A
    53it [10:42,  8.60s/it][A
    54it [10:47,  7.74s/it][A
    55it [10:53,  7.07s/it][A
    56it [10:59,  6.78s/it][A
    57it [11:04,  6.32s/it][A
    58it [11:08,  5.72s/it][A
    59it [11:15,  5.83s/it][A
    60it [11:20,  5.75s/it][A
    61it [11:24,  5.31s/it][A
    62it [11:31,  5.56s/it][A
    63it [11:37,  5.78s/it][A
    64it [11:43,  6.01s/it][A
    65it [11:48,  5.49s/it][A
    66it [11:54,  5.69s/it][A
    67it [12:00,  5.79s/it][A
    68it [12:05,  5.56s/it][A
    69it [12:10,  5.55s/it][A
    70it [12:15,  5.19s/it][A
    71it [12:21,  5.40s/it][A
    72it [12:24,  4.67s/it][A
    73it [12:28,  4.73s/it][A
    74it [12:45,  8.27s/it][A
    75it [12:57,  9.24s/it][A
    76it [13:06,  9.34s/it][A
    77it [13:13,  8.73s/it][A
    78it [13:24,  9.43s/it][A
    79it [13:33,  9.05s/it][A
    80it [13:43,  9.37s/it][A
    81it [13:49,  8.32s/it][A
    82it [13:57,  8.32s/it][A
    83it [14:06,  8.68s/it][A
    84it [14:15,  8.63s/it][A
    85it [14:22,  8.04s/it][A
    86it [14:29,  7.95s/it][A
    87it [14:36,  7.61s/it][A
    88it [14:44, 10.05s/it]
     50%|█████     | 1/2 [14:45<14:45, 885.01s/it]
    0it [00:00, ?it/s][A
    2it [00:00, 17.78it/s][A
    4it [00:00,  5.88it/s][A
    5it [00:04,  1.17s/it][A
    6it [00:08,  2.15s/it][A
    7it [00:14,  3.20s/it][A
    8it [00:40,  9.99s/it][A
    9it [01:08, 15.57s/it][A
    10it [01:34, 18.69s/it][A
    11it [02:02, 21.36s/it][A
    12it [02:30, 23.27s/it][A
    13it [02:59, 24.97s/it][A
    14it [03:25, 25.44s/it][A
    15it [03:45, 23.84s/it][A
    16it [04:01, 21.49s/it][A
    17it [04:17, 19.61s/it][A
    18it [04:46, 22.59s/it][A
    19it [05:02, 20.61s/it][A
    20it [05:34, 24.01s/it][A
    21it [05:43, 19.65s/it][A
    22it [05:54, 16.91s/it][A
    23it [06:04, 14.80s/it][A
    24it [06:13, 13.21s/it][A
    25it [06:22, 11.82s/it][A
    26it [06:32, 11.15s/it][A
    27it [06:40, 10.44s/it][A
    28it [07:07, 15.23s/it][A
    29it [07:14, 12.80s/it][A
    30it [07:21, 11.17s/it][A
    31it [07:51, 16.76s/it][A
    32it [08:17, 19.48s/it][A
    33it [08:35, 19.11s/it][A
    34it [08:52, 18.49s/it][A
    35it [09:05, 16.70s/it][A
    36it [09:21, 16.48s/it][A
    37it [09:28, 13.87s/it][A
    38it [09:29,  9.93s/it][A
    39it [09:38,  9.58s/it][A
    40it [10:03, 14.35s/it][A
    41it [10:30, 18.05s/it][A
    42it [10:58, 20.92s/it][A
    43it [11:11, 18.55s/it][A
    44it [11:29, 18.62s/it][A
    45it [11:45, 17.69s/it][A
    46it [12:01, 17.25s/it][A
    47it [12:18, 17.02s/it][A
    48it [12:31, 15.91s/it][A
    49it [12:46, 15.74s/it][A
    50it [13:00, 15.08s/it][A
    51it [13:11, 13.93s/it][A
    52it [13:21, 12.81s/it][A
    53it [13:30, 11.41s/it][A
    54it [13:40, 11.00s/it][A
    55it [13:50, 10.74s/it][A
    56it [14:05, 12.17s/it][A
    57it [14:13, 10.75s/it][A
    58it [14:24, 10.85s/it][A
    59it [14:35, 10.88s/it][A
    60it [14:47, 11.18s/it][A
    61it [15:00, 11.95s/it][A
    62it [15:11, 11.65s/it][A
    63it [15:22, 11.40s/it][A
    64it [15:25,  8.81s/it][A
    65it [15:34,  8.80s/it][A
    66it [15:47, 10.18s/it][A
    67it [15:55,  9.53s/it][A
    68it [16:03,  8.98s/it][A
    69it [16:10,  8.34s/it][A
    70it [16:17,  7.99s/it][A
    71it [16:24,  7.72s/it][A
    72it [16:31,  7.54s/it][A
    73it [16:36,  6.79s/it][A
    74it [16:42,  6.55s/it][A
    75it [16:52,  7.58s/it][A
    76it [17:04,  8.93s/it][A
    77it [17:12,  8.71s/it][A
    78it [17:21,  8.64s/it][A
    79it [17:36, 10.58s/it][A
    80it [17:51, 12.05s/it][A
    81it [18:11, 14.23s/it][A
    82it [18:33, 16.66s/it][A
    83it [18:45, 15.29s/it][A
    84it [19:15, 19.84s/it][A
    85it [19:41, 21.51s/it][A
    86it [20:13, 14.11s/it]
    100%|██████████| 2/2 [34:59<00:00, 1049.80s/it]
    


```python
naver_emo
# naver_mood
# naver_daily

# twitter_emo
# twitter_mood
twitter_daily.to_excel("twitter_daily_2.xlsx")
```
