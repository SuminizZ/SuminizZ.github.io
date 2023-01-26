---
layout: post
title : "[Minimal Mistakes] - Masthead, Header"
date: 2021-10-18 00:00:00
categories : [github-blogdev]
tag : [Github, Jekyll, minimal-mistakes]
tags : [Github, Git]
toc : true
toc_sticky : true
---
<br/>

## Masthead
<br/>

- 가장 상단의 로고와 카테고리 navigation 이 있는 공간이다.
<br/>

### **- Logo**
<br/>

- ```_config.yml``` 의 **# Site Settings** 항목에 **logo** 에서 변경할 수 있다
- ```assets/images/``` 에 원하는 로고 파일을 저장한 뒤, 아래와 같이 해당 이미지의 경로을 logo 값에 넣어준다. <br>
<br>
```
logo : "/assets/images/<name>.png"
```
<br/>

### **- Masthead 전반적인 디테일 수정**
<br/>

- ```_sass\minimal-mistakes\_masthead.scss``` 에서 14번째 줄 근처 **&__inner-wrap** 내부의 값을 조절해주면 된다.
- 아래처럼 **font-size, font-weight** 값을 조절해 masthead 공간에 포함된 text 의 폰트 사이즈, 굵기를 지정해줄 수 있다.
- padding 값을 조절해 전체적인 크기를 수정할 수도 있다.<br>
<br>
```css
  &__inner-wrap {
    @include clearfix;
    margin-left: auto;
    margin-right: auto;
    padding: 0.5em;
    max-width: 100%;
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-pack: justify;
    -ms-flex-pack: justify;
    justify-content: space-between;
    font-family: $sans-serif-narrow;
    
    font-size: $type-size-4;
    font-weight: bold;
```
<br/>

## Header
<br/>

### **- Header 에 이미지 삽입하기**
<br/>

- ```./index.html``` 에 아래 코드를 삽입한다.<br>
<br>
```md
header:
  overlay_image: assets\images\galaxy2.jpg
  overlay_filter: 0.5
```


