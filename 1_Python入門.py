# データ型の確認
hoge = 1.234
print(type(hoge))

# 二次元配列
arry = [1, 2, 3, 4, 5]
print(arry)
print('長さ:' + str(len(arry)))
print('最初の要素:' + str(arry[0]))
# スライシング
print('0から2番目の要素（2番目は含まない！）:' + str(arry[0:2]))
print('1番目から最後までの要素:' + str(arry[1:]))
print('最初から2番目の要素（2番目は含まない！）:' + str(arry[:2]))

# ディクショナリ
dic = {'x': 100, 'y':200}
print(dic['x'])
    
# クラス
class cls:
    # コンストラクタ
    def __init__(self, mes):
        self.message = mes
    def display(self):
        print(self.message)

cls_test = cls('hello!')
cls_test.display()


import numpy as np

ndarry1 = np.array([1,2,3])

print(ndarry1)
print(type(ndarry1))

ndarry2 = np.array([2,4,6])

print(ndarry1 - ndarry2)
print(ndarry1 / 2)

ndarry3 = np.array([[1, 2], [3, 4]])
ndarry4 = np.array([[2, 4], [6, 8]])

print(ndarry3 + ndarry4)
print(ndarry3[1][0])

print(ndarry3 > 2)
print(ndarry3[ndarry3 > 2])

import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.image import imread
import numpy as np

x = np.linspace(0, 20, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='cos')
plt.xlabel('x-label') # x軸のラベル
plt.ylabel('y-label') # y軸のラベル
plt.title('sin & cos graph')
img = imread('bird.png')
plt.imshow(img)
plt.legend()
plt.show() 
