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
