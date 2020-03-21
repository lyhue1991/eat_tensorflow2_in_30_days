# -*- coding: utf-8 -*-
# ## 推送主分支

# !git config --global user.email "lyhue1991@163.com"
# 出现一些类似 warning: LF will be replaced by CRLF in <file-name>. 可启用如下设置。
# !git config --global core.autocrlf false
# 配置打印历史commit的快捷命令
# !git config --global alias.lg "log --oneline --graph --all"

# !git init

# !git add  ./data/*  *.md *.py

# !rm -rf *.html

# +
# #!git rm --cached  push-to-github.md
# -

# !git commit -m"add chapter6"

# +
# #!git remote rm origin 

# +
# #!git remote add origin https://github.com/lyhue1991/eat_tensorflow2_in_30_days

# +
# #!git pull origin master 
# -

# !git push  origin master 

# ## 创建pages分支

# !git checkout -b gh-pages

# !git branch -m pages gh-pages

# !git rm --cached -r *.md

# !git clean -df
# !rm -rf *.md

# !cp -r _book/* .

# !git add .

# !git reset

# !git pull origin gh-pages

# !git commit -m 'add gh-pages'

# !git push -u origin gh-pages

# !git checkout pages

# ## 更新命令

# !git checkout master

# !git add ./data/*  *.md *.py
# !git commit -m "revise readme"

# !git push -u origin master

# !gitbook build

# !git checkout gh-pages

# !cp -r _book/* .

# !git add .
# !git commit -m "revise readme"
# !git push -u origin gh-pages
# !git checkout master
