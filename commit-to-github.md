```python
!git config --global user.email "lyhue1991@163.com"

# 出现一些类似 warning: LF will be replaced by CRLF in <file-name>. 可启用如下设置。
!git config --global core.autocrlf false

# 配置打印历史commit的快捷命令
!git config --global alias.lg "log --oneline --graph --all"
```

```python
!git init
```

```python
!git add  ./data/* ./*.md ././*.md
```

```python
!git rm -r ./.ipynb_checkpoints
```

```python
!git commit -m"add readme"
```

```python
!git remote add origin https://github.com/lyhue1991/lyhue1991.github.io
```

```python
!git push origin master 
```
