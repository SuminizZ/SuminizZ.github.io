---
layout: post
title : "[Minimal Mistakes] - 폰트 변경, 프로필 사진 설정"
date: 2021-10-18 00:00:00
categories : [github-blogdev]
tag : [Github, Jekyll, minimal-mistakes]
tags : [Github, Git]
toc : true
toc_sticky : true
---

## 폰트 변경 및 크기 조절 (예시 폰트 : 리디바탕)
<br/>

### **폰트 변경**
- 일단 woff 형식으로 된 원하는 폰트를 다운받아 ```./assets/fonts``` 폴더에 넣어준다.
- ```./assets/css/main.css``` 의 내부에 아래 코드를 추가한다.<br>
(**font-family** 에 원하는 폰트 이름을 넣고, **src : url()**안에 해당 폰트의 경로를 넣어준다.)<br>
<br>
```css
@font-face {
    font-family: 'RIDIBatang';
    src: url('/assets/fonts/RIDIBatang.woff') format('woff');
    font-weight: normal;
    font-style: normal;
}
```
- ```_sass/minimal-mistakes/_variables.scss``` 파일의 **system typefaces > sans-serif** 항목을 "RIDIBatang" 으로 변경한다. <br>
<br>
```css
$sans-serif: "RIDIBatang", -apple-system, BlinkMacSystemFont, "Roboto", "Segoe UI",
  "Helvetica Neue", "Lucida Grande", Arial, sans-serif !default;
```

<br/>

### **폰트 크기 조절**
- ```_sass/_reset.scss``` 에서 아래 코드의 **font-size** 를 조절한다. 대략 14~17px 정도가 적당한 듯하다. 창 크기 조절시에 폰트 크기 변경되는 효과를 제거해줬다. <br>

    ```css
    /* ==========================================================================
    STYLE RESETS
    ========================================================================== */
    
    *{ box-sizing: border-box; }
    
    html {
    /* apply a natural box layout model to all elements */
    box-sizing: border-box;
    background-color: $background-color;
    font-size: 15px;
    @include breakpoint($medium) {
        font-size: 17px;
    }
    @include breakpoint($large) {
        font-size: 17px;
    }
    @include breakpoint($x-large) {
        font-size: 17px;
    }
    -webkit-text-size-adjust: 100%;
    -ms-text-size-adjust: 100%;
    }
    ```
<br/>

## 프로필 사진 업로드 및 프로필 Figure 조절
<br/>

### **업로드**
- ```assets/images``` 폴더에 원하는 프로필 사진을 추가한다
- ```_config.yml``` 폴더의 **Site Author** 항목의 **avatar** 값으로 프로필 사진의 위치를 기입해준다. <br>
<br>
```python
avatar : "/assets/avatar.jpg"
```

<br/>

### **프로필 Figure 조절**
- ```_sass/_sidebar.scss``` >  **Author profile and links** 에서 **author_avatar > img** 섹션의 값을 조절한다. 
- **max_width** 로 사진의 크기를 변경한다(내 기준 160-180px 정도가 적당한 거 같다.)
- **border_radius** 는 이미지의 각진 정도인데, 50% 가 max(완전 곡선) 이고 그보다 낮아지면 각이 생긴다 0% 에 가까울 수록 직사각형이 된다. 
- 비율은 원본 사진 비율 그대로 유지되는 거 같다.
- **author_avatar > padding** 값을 조절하면 프로필 사진의 위치 및 여백 크기도 바꿀 수 있다. <br>
<br>
```css
.author__avatar {
  display: table-cell;
  vertical-align: top;
  width: 36px;
  height: 36px;
  padding-top: 20px;
  padding-right: 30px;
  padding-bottom: 30px;
```

