
# 阿里音乐流行趋势预测大赛，第一季第二个数据集（100个用户）
## 程序

## 依赖的运行环境
* Ubuntu
* python 2.7
* 可以安装pip，然后用pip安装python的libs
* ipython （python开发环境）
* matplotlib (用来画图表)
* numpy (用来作数据处理)
* pandas (用来作数据处理)
* scikit-learn （用来作学习) [安装方法](http://www.bogotobogo.com/python/scikit-learn/scikit-learn_install.php)  

## data文件夹
放置mars_tianchi_songs.csv、mars_tianchi_user_actions.csv两个文件

## 运行ipython
在shell下面运行ipython
```
ipython --pylab
```
## 思路
思路是stl分解出每周、每月的周期波动，然后用arima学习大的趋势
