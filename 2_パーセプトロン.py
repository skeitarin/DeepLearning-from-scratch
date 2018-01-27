# x:input
# w:weight
# y:output
# b:bias

import numpy as np

class perceptron:
    def __init__(self, w, b):
        self.w, self.b = w, b
    def fire(self, x):
        x = np.sum(x * self.w) + self.b # NumPyのブロードキャスト機能を使用
        if(x > 0):
            return 1
        else:
            return 0 

f_f = np.array([0, 0])
f_t = np.array([0, 1])
t_f = np.array([1, 0])
t_t = np.array([1, 1])

print('-- AND GATE --')
AND_gate = perceptron(np.array([0.5, 0.5]), -0.6)
print(AND_gate.fire(f_f)) # -> 0
print(AND_gate.fire(f_t)) # -> 0
print(AND_gate.fire(t_f)) # -> 0
print(AND_gate.fire(t_t)) # -> 1

print('-- NAND GATE --')
NAND_gate = perceptron(np.array([-0.5, -0.5]), 0.6)
print(NAND_gate.fire(f_f)) # -> 1
print(NAND_gate.fire(f_t)) # -> 1
print(NAND_gate.fire(t_f)) # -> 1
print(NAND_gate.fire(t_t)) # -> 0

print('-- OR GATE --')
OR_gate = perceptron(np.array([0.5, 0.5]), -0.4)
print(OR_gate.fire(f_f)) # -> 0
print(OR_gate.fire(f_t)) # -> 1
print(OR_gate.fire(t_f)) # -> 1
print(OR_gate.fire(t_t)) # -> 1

print('-- XOR GATE --')
def XOR_gate(x):
    AND_gate = perceptron(np.array([0.5, 0.5]), -0.6)
    NAND_gate = perceptron(np.array([-0.5, -0.5]), 0.6)
    OR_gate = perceptron(np.array([0.5, 0.5]), -0.4)

    s1 = NAND_gate.fire(x)
    s2 = OR_gate.fire(x)
    y = AND_gate.fire(np.array([s1, s2]))
    return y

print(XOR_gate(f_f)) # -> 0
print(XOR_gate(f_t)) # -> 1
print(XOR_gate(t_f)) # -> 1
print(XOR_gate(t_t)) # -> 0