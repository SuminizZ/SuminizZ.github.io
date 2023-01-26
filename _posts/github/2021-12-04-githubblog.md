---
title : "[Minimal Mistakes] - Favicon 등록, Scroll Bar 커스텀"
categories : 
    - Blog Dev
tag : [Github, Jekyll, minimal-mistakes]
toc : true
toc_sticky : true
---

## Favicon 등록하기

- 원하는 사진 파일을 준비한다.
- [realfavicongenerator](https://realfavicongenerator.net/) 에 들어가 **select your favicon image** 버튼을 누른 후 해당 사진을 업로드한다. 
- 페이지 최하단의 **Generate your Favicons and HTML code** 버튼을 누른다.
- **Favicon Package** 를 다운로드 후 압축해제하고 전체 파일을 github blog 의 최상위 루트폴더에 저장한다. (다른 경로에 넣으면 안 됨)
- HTML 코드를 복사해서 ```_includes/head/custom.html``` 에 복붙해준다. (단, 이때 href 위치는 favicon package folder 의 relative path 로 변경해줘야 한다.)
<br/>

    ```css
        <link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
        <link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png">
        <link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png">
        <link rel="manifest" href="site.webmanifest">
        <link rel="mask-icon" href="safari-pinned-tab.svg" color="#5bbad5">
        <meta name="msapplication-TileColor" content="#da532c">
        <meta name="theme-color" content="#ffffff">
    ```

## Scroll Bar 커스텀

- ```_layouts/default.html``` 에 아래 코드를 복붙한다.
<br/>
    ```css
        <style> 
            ::-webkit-scrollbar{width: 13px;}
            ::-webkit-scrollbar-track {background-color:#4b4f52; border-radius: 16px;}
            ::-webkit-scrollbar-thumb {background-color:#5e6265; border-radius: 16px;}
            ::-webkit-scrollbar-thumb:hover {background: #ffd24c;}
            ::-webkit-scrollbar-button:start:decrement,::-webkit-scrollbar-button:end:increment 
            {
                width:10px;height:12px;background:transparent;}
            } 
        </style>
    ```

- width, height, background (scroll bar 색), border-radius (bar 모서리 둥근 정도) 등을 조절하면 된다
