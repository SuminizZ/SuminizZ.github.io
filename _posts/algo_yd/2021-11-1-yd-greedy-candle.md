---
title : "[Year-Dream 스터디 12주차 : 그리디, Python] - 도도새의 절약 정신"
categories : 
    - Year-Dream-L2D
tag : [Year-Dream, L2D, Python, 그리디, Greedy, Algorithm]
toc : true
---
## **문제**
이어드림 12주차 알고리즘 스터디 문제 : 도도새의 절약 정신(상)

<img src="https://user-images.githubusercontent.com/92680829/139600828-5f9de4c2-b379-411b-90d7-194b97e8ea9b.png" />


## **Solution**

```python
import sys
input = sys.stdin.readline

n, k = map(int, input().split())

time = []
for _ in range(n):
    time.append(int(input()))
    
total = 0
gap = []
for i in range(n-1):
    cur_gap = time[i+1] - time[i]
    gap.append(cur_gap)        
    total += cur_gap

total = total + 1       # 마지막 촛불

gap.sort(key = lambda x : x, reverse=True)

off = gap[:k-1]

for i in range(len(off)):
    total -= off[i] - 1
    
print(total)

```

## **풀이과정 및 느낀점**
1. 우선 방문하는 모든 시간 사이의 간격을 total 값에 다 더한다.
2. 간격이 긴 순서대로 내림차순으로 정렬하여(그리디) 그 중 k-1 (1번째 손님에서 이미 1번 사용) 개까지만 slicing 해서 최종적으로 난로를 끌 애들만 off 리스트에 모은다.
3. off 리스트를 for 문으로 돌면서 모든 간격에 -1 을 해서 total 값에서 빼준다. (간격 중 1시간은 실제로 손님이 도도새 집에 머무르는 시간, 난로 끄지 않음)
