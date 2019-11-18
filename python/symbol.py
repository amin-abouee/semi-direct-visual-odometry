import sympy as sym

sym.init_printing()
fx, fy, cx, cy = sym.symbols('fx fy cx cy')
matrix = sym.Matrix([[fx, 0.0, cx], 
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])

print(matrix.det())
print(matrix.inv())