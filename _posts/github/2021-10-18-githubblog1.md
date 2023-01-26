---
layout: post
title : "[Minimal Mistakes] - 셋업환경 설정, 로컬 블로그 생성"
date: 2021-10-18 00:00:00
categories : [github-blogdev]
tag : [Github, Jekyll, minimal-mistakes]
tags : [Github, Git]
toc : true
toc_sticky : true
---

<br/>

## 셋업환경 설치하기
- GithubDesktop 을 다운로드 받는다
- GithubDesktop 을 시작할 때 블로그용 Repository(suminiz.github.io) 를 clone 해 저장할 Project 폴더를 만든다. 
```C:\ProgramData\LG\GitHubDesktop\Project\suminiz_github_blog```
- Visual Studio 에서 위 Project 폴더를 연다. 앞으로 Github.io 와 관련된 거의 모든 작업은 VSCode에서 수행된다.
- Typora 설치, 역시 위 Project 폴더를 열어준다. (개인적으로 이미지 drag & paste하는 경우를 제외하고는 Typora 는 Visual Studio 기능으로 충분히 대체가능한 거 같다. 심지어 이미지 업로드도 github issues 를 통한 방식이 더 나음)

<br/>

## 로컬 블로그 생성하기 with Ruby
- Ruby 설치. 이때, 아래 **체크사항**을 꼭 주의하자. <br> 
1. ```Add Ruby executables to your PATH``` 에 체크 <br>
2. ```MSYS2 development toolchain``` 에도 체크
- cmd 창에서 Jekyll 과 Bundler 설치 <br>
<br>
``` python
gem install jekyll
gem install bundler
```

- 설치가 완료되면, Powershell 을 githubDesktop 의 블로그 폴더에서 시작 ```C:\ProgramData\LG\GitHubDesktop\Project\suminiz_github_blog\suminiz.github.io``` <br>
(shift 누른 상태로 우클릭 + powershell 시작하기) <br>
<br>
    ```python
    bundle install
    bundle add webrick
    bundle exec jekyll serve
    ```

- 로컬환경의 Server address 발급해주면 해당 주소로 가서 즉각적으로 블로그 셋업 변경을 확인한다
- Error : TimeZone 셋업하면 **`require': cannot load such file -- tzinfo (LoadError)** 이런 에러가 뜨는데, 그럼 _config.yml 에서 timeZone 초기화해준다. 
- Error 관련 사항은 아래 블로그의 도움을 받았다. <br>
> https://imsejin.github.io/articles/jekyll/create-jekyll-blog



