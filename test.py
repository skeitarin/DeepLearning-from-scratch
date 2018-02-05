import numpy as np

def f(x):
    return x**2
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # xと同じ型の配列を生成

    for i in range(x.size):
        print("x:" + str(x) + ", i:" + str(i))
        tmp = x[i]
        # f(x+h)の計算
        x[i] = tmp + h
        f_1 = f(x)
        
        # f(x-h)の計算
        x[i] = tmp - h
        f_2 = f(x)

        grad[i] = (f_1 - f_2) / (2 * h) # 偏微分（x[i]に限りなく0に近い値を増減し傾きを計算。その他は定数化）
        x[i] = tmp
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
        print("grad, x | " + str(grad) + ", " + str(x))
    print("-----------")
    return x

print(gradient_descent(f, np.array([10.0]), lr=1.0, step_num=100))