---
layout: post
title : "[Minimal Mistakes] - 404 Not Found, 게시글 링크 커스텀 + 스킨 색 커스텀"
date: 2021-10-18 00:00:00
categories : [github-blogdev]
tag : [Github, Jekyll, minimal-mistakes]
tags : [Github, Git]
toc : true
toc_sticky : true
---
<br/>

## Error : 404 Not Found
<br/>

- 404 Error 페이지를 커스터마이징하기 위한 방법이다.
- 우선 ```_pages``` 폴더에 **404.md** 파일을 생성한다.
- ```test/_pages/404.md``` 에 있는 전문을 처음 생성한 404.md 파일에 붙여넣는다.
- Customizing 을 하고 싶으면 404 에러와 관련된 이미지를 찾아 이미지 업로드를 하거나, Text 내용을 수정할 수도 있다.
<br>
(아래는 내가 만든 404 페이지이다)<br>
<br>
~~~md
---
title: "Page Not Found"
excerpt: "Page not found. Your pixels are in another canvas."
sitemap: false
permalink: /404.html
<img src="https://i.stack.imgur.com/cGqdX.png" width="1000px"/>
<br>
Sorry, but the page you were trying to view does not exist. :( 
Try another one
---
~~~
<br/>

## 게시글 링크 커스텀
<br/>

### **- 게시글 링크 언더라인 제거**

<br/>

- ```_sass/minimal-mistakes/_base.css``` 로 가서, **/* links */** 항목으로 이동한다 (127번재 줄)
- **a** 태그 안에 **text-decoration: none;** 을 추가한다. <br>
<br>
```css
/* links */
a {
  text-decoration: none;
  
  &:focus {
    @extend %tab-focus;
  }
  
  &:visited {
    color: $link-color-visited;
  }
  
  &:hover {
    color: $link-color-hover;
    outline: 0;
  }
}
```

<br/>


### **- 게시글 색 변경**
<br/>

- ```/sass/minimal-mistakes/skins/(내스킨).scss``` 에서 변경할 수 있다. 나는 Contrast 를 적용하고 있기 때문에 아래 코드는 모두 해당 스킨의 소스코드이다.
- 내 스킨은 ```_config.yml``` 의 **minimal-mistakes-skin** 에서 확인할 수 있다.
- 우선, **/* Colors */** 항목의 **link-color** 색을 원하는 색으로 변경해주고, 아래의 코드를 적당히 아래에 붙여넣어주면 된다.<br>
```css
.page__title {
  color: $link-color;
}
```

- 이곳에서 스킨의 모든 (background, masthead, nav, footer-background-color, page-content) 색을 변경할 수 있다.
