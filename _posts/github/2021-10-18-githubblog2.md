---
layout: post
title : "[Minimal Mistakes] - 게시글 포스팅, 이미지 업로드"
date: 2021-10-18 00:00:00
categories : [github-blogdev]
tag : [Github, Jekyll, minimal-mistakes]
tags : [Github, Git]
toc : true
toc_sticky : true
---

## 포스팅

<br/>

- _posts 폴더를 생성하고, md 형식의 파일을 생성한다 (```"2021-10-16-title.md"```)
- "--- ---" 내부에 layout, title, categories 등의 정보를 입력해준다
```md
---
layout : single
title : "Github 블로그 셋업하기"
categories : Blog-Setup
toc : true
---
```
- **toc: true** 는 h tag 의 위계에 따라 포스팅의 목차를 자동으로 생성해준다

## 이미지 업로드
<br/>

### **Github Issues 로 경로 발급받기**

1. 웹에서 찾은 이미지는 바로 이미지 주소를 복사해서 css 형식의 ```<img src="주소">``` 형식으로 VSCode 창에 입력한다.<br>
<br>
```
<img src="https://cdnweb01.wikitree.co.kr/webdata/editor/202004/07/img_20200407162305_1f42c686.webp" width="300px"/>
```
2. 로컬 이미지 혹은 캡쳐한 이미지는 복사해서 아래와 같은 **Github - New Issues** 에 붙여넣는다 <br>
<br/>
<img src="https://user-images.githubusercontent.com/89923538/137593226-3334605a-83eb-4409-a06f-db08e6233ff1.png" width="650px"/> <br>
<br/>

* 잠시 기다리면 새로운 이미지 주소를 발급해준다
- 해당 주소를 복사해 1. 의 방식대로 ```<img src="주소">``` 형식으로 VSCode 창에 입력한다.

