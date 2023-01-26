---
layout: post
title : "[Minimal Mistakes] - 카테고리, 태그 지정 + Left Sidebar Navigation"
date: 2021-10-18 00:00:00
categories : [github-blogdev]
tag : [Github, Jekyll, minimal-mistakes]
tags : [Github, Git]
toc : true
toc_sticky : true
---
<br/>

## 블로그 상단 메뉴바 카테고리 및 태그 추가
- ```_config.yml``` 의 **Archives > jekyll-archives** 항목의 전체를 주석 해제해준다. 
- ```./_pages``` 폴더에(없으면 생성) **category-archive.md** 파일과 **tag-archive.md** 파일을 생성해준다. 
- 각각의 파일에 아래의 내용을 복붙해준다.
- 카테고리 파일 (category-archive.md)
```md
---
title : "Category"
layout : categories
permalink : /categories/
author_profile : true
sidebar_main : true
---
```

- 태그 파일 (tag-archive.md)
```md
---
title : "Tag"
layout : tags
permalink : /tags/
author_profile : true
sidebar_main : true
---
```
<br/>

## 왼쪽 Sidebar 카테고리 설정 (Sidebar Navigation)
- 이 파트는 아래 주소의 블로그에 매우 잘 설명되어 있다. <br>
    **<https://ansohxxn.github.io/blog/category>**
- 단, 주의할 점은 posting 할 때 상단에 적는 **[카테고리 이름]**이 ```_pages/categories/``` 폴더에 있는 md 파일의 **sites.categories.[카테고리이름]** 과 일치해야 한다는 것이다.
- 이 두 파일의 역할은 특정 카테고리에 해당하는 포스팅들만 뽑아서 해당 카테고리가 연결된(permalink) 페이지(categories/[카테고리]) 에서 보여주는 것이다.
- nav_list_main 의 역할은 왼쪽 사이드바의 각각의 카테고리를 클릭하면 연결된 링크(href="/categories/[카테고리]", should be same as permalink) 로 이동하게 해주는 것이다.
- nav_list_main 파일에서 특정 카테고리를 인식하는 방식(**if category[0] == "~~"**)은 ```_pages/categories/``` 의 각 카테고리의 md 에 지정된 카테고리 이름이다. 

1. ```_posts/~~~.md```
    ~~~md
    ---
    title : "[Github Blog] - title"
    categories : 
        - [카테고리 이름]
    tag : [Github, Jekyll, minimal-mistakes]
    toc : true
    ---
    ~~~

