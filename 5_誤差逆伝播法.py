# 乗算レイヤー
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# 加算レイヤー
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        return x + y
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

mul_apple_lyaer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_lyaer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_lyaer.backward(dapple_price)

print(dapple, dapple_num, dtax)

# 図5-17の実装（p149）
apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
apple_orange_add_layer = AddLayer()
tax_mul_layer = MulLayer()

apple_num = 2
apple_price = 100
orange_num = 3
orange_price = 150
tax = 1.1

# forward
apple_sum_price = apple_mul_layer.forward(apple_num, apple_price)
orange_sum_price = orange_mul_layer.forward(orange_num, orange_price)
apple_orrange_sum_price = apple_orange_add_layer.forward(apple_sum_price, orange_sum_price)
sum_price = tax_mul_layer.forward(apple_orrange_sum_price, tax)
print(sum_price)

dsum_price = 1
# backward
dapple_orrange_sum_price, dtax = tax_mul_layer.backward(dsum_price)
dapple_sum_price, dorange_sum_price = apple_orange_add_layer.backward(dapple_orrange_sum_price)
dapple_num, dapple_price = apple_mul_layer.backward(dapple_sum_price)
dorange_num, dorange_price = orange_mul_layer.backward(dorange_sum_price)
print(dapple_num, dapple_price, dorange_num, dorange_price, dtax)