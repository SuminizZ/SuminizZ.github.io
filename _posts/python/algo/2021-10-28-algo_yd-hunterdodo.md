---
title : "[Year-Dream 스터디 12주차 : 동적계획법, Python] - 사냥꾼 도도새"
layout : post
categories : 
    - [python-algo]
tag : [Year-Dream, L2D, Python, DP, 동적계획법, Dynamic Programming, Algorithm]
toc : true
---

<br/>

## 이어드림 12주차 알고리즘 스터디 문제 : 사냥꾼 도도새(중) 

<br/>

<img src="https://user-images.githubusercontent.com/92680829/139163117-0474d80e-b813-4d23-a078-0388fdc91dfb.png" width="720px"/>


<br/>

## **Solution**

<br/>

### DP Memoization 활용 (Bottom-up)

<br/>

```python
import sys
input = sys.stdin.readline

n = int(input())
heights = list(map(int, input().split()))

DP = [float('inf') for _ in range(n+1)]        # k 번째 거미까지 필요한 총알 개수 저장
DP[0] = 1
DP[1] = 1

for i in range(2, n+1):
    for j in range(1, i):
        if heights[i-1] == heights[j-1] - 1:
            DP[i] = DP[j]
            heights[j-1] = -1   
            break                           # 먼저 쏜 총알에 의해 이미 거미는 떨어진다.
    else:                                   # for - else 문
        DP[i] = min(DP[i], max(DP[:i]) + 1)
    
    # print(DP[i])
        
print(max(DP))
```

<br/>

## **풀이과정 및 느낀점**

<br/>

1. k번재 거미는 이 전의 거미들 중 자기보다 1칸 더 높은 거미들 중 첫 번째 거미의 총알 개수를 이어받는다.
2. 만약 없다면, 새로운 총알이 필요하다는 의미이다. 코드 상으로는, for 문을 모두 도는 과정에서 한 번도 if 문을 타지 못하기 때문에 else 문에 걸려 이전 거미들의 총알 개수 중 가장 큰 값 + 1 이 저장된다.
    - **주의** 이 else 문을 2번째 for 문 안으로 넣게 되면 매번 slicing 과정 + 조회 과정을 거치게 되기 때문에 5번 test case 에서 시간초과가 뜬다.
