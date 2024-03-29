import sympy as sym


# sym.init_printing()
fx, fy, cx, cy = sym.symbols('fx fy cx cy')
K = sym.Matrix([[fx, 0.0, cx], 
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])

tx, ty, tz = sym.symbols('tx ty tz')
t = sym.Matrix([[tx], [ty], [tz]])

R = sym.MatrixSymbol('R', 3, 3)

X, Y, Z = sym.symbols('X Y Z')
point = sym.Matrix([[X], [Y], [Z]])

sym.pprint(K * (R * point + t), use_unicode=True)

# sym.pprint(matrix.det(), use_unicode=True)
# sym.pprint(matrix.inv(), use_unicode=True)

tx, ty, tz = sym.symbols('tx ty tz')

K = sym.Matrix([[fx, 0.0, cx, 0.0], 
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

E = sym.Matrix([[1.0, 0.0, 0.0, tx], 
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0]])

P = K * E
sym.pprint(K, use_unicode=True)
sym.pprint(E, use_unicode=True)
sym.pprint(P, use_unicode=True)
print(sym.latex(K, mode='equation'))
print(sym.latex(E, mode='equation'))
print(sym.latex(P, mode='equation'))

x, y, z, z2 = sym.symbols('x y z z2')
point = sym.Matrix([[x], [y], [z], [1.0]])
sym.pprint(point, use_unicode=True)
res = P * point
sym.pprint(res, use_unicode=True)
print(sym.latex(res, mode='equation'))

x, y, z, z2 = sym.symbols('x y z z2')
first = sym.Matrix([[fx/z, 0.0, -fx * (x/z2)], 
                     [0.0, fy/z, -fy * (y/z2)]])
sym.pprint(first, use_unicode=True)

second = sym.Matrix([[1, 0, 0, 0, z, -y], [0, 1, 0, -z, 0, x], [0, 0, 1, y, -x, 0]])
sym.pprint(second, use_unicode=True)

final = first * second
sym.pprint(final, use_unicode=True)
print(sym.latex(final, mode='equation'))

# from sympy.mpmath import *

# print(sym.__version__)
# poly, err = sym.mpmath.chebyfit(cos, [-pi, pi], 5, error=True)
