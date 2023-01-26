---
title : "[Year-Dream 스터디 12주차 : 동적계획법, Python] - 거스름돈 주기"
categories : 
    - Year-Dream-L2D
tag : [Year-Dream, L2D, Python, DP, 동적계획법, Dynamic Programming, Algorithm]
toc : true
---
## **문제**
이어드림 12주차 알고리즘 스터디 문제 : 거스름돈 주기(상) 

<img src="https://user-images.githubusercontent.com/92680829/139757582-81b2994f-895b-48a9-a2a7-f084ce329451.png" width="720px"/>


## **Solution**
### 정답 코드

```python
import sys
# from collections import defaultdict
input = sys.stdin.readline

n = int(input())
coins = list(map(int, input().split()))
# coins.sort(reverse=True)
change = int(input())
 
DP = [0 for _ in range(change+1)]

DP[0] = 1

for c in coins:                 
    for i in range(1, change+1):        # 특정 동전만 사용해서 지불하는 경우
        if i - c >= 0:
            DP[i] += DP[i - c]
            
print(DP)

```

### 오답 코드
```python
for i in range(1, change+1):
    for c in coins:           # 동일한 케이스가 누적해서 더해짐
        if i - c >= 0:
            DP[i] += DP[i-c]

```

## **풀이과정 및 느낀점**
이 문제의 관건은 DP에 거스름돈 주는 경우의 수가 누적해서 더해지는 걸 방지하는 것이다.
<br/>
처음에는 오답 코드처럼 문제를 풀었는데 이렇게 하면 특정 금액 자체가 기준이 돼고(첫 번째 for 문) 거스름돈 경우의 수 계산은 이후에 이뤄져서(두 번째 for 문) 거스름돈을 10 -> 50 한 경우와 50 -> 10 준 경우가 중복돼서 계산된다.
<br/>
따라서, 위의 정답코드처럼 for 문의 순서를 뒤바꿔 각 단위의 거스름돈을 사용해 지불 가능한 금액들을 먼저 DP 에 다 저장하는 식으로 수정하게 되면 중복계산없이 DP를 완성할 수 있다.