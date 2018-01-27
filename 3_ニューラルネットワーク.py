import numpy as np
# 活性化関数

# ステップ関数
def step_function(x):
    if(x > 0):
        return 1
    else:
        return 0
    # ↑引数のxは実数のみ、ndarrayなどは渡せない

x1 = 0.1
x2 = np.array([1, -1])

print(step_function(x1))
# print(step_function(x2)) #-> Error

def step_function2(x):
    x2 = x > 0
    return x2.astype(np.int)

print(step_function2(x2))