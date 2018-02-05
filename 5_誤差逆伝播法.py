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

apple = 100
apple_num = 2
tax = 1.1

mul_apple_lyaer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_lyaer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_lyaer.backward(dapple_price)

print(dapple, dapple_num, dtax)
