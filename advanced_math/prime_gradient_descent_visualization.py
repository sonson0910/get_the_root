import numpy as np
import matplotlib.pyplot as plt

# Tạo lưới điểm (x,y)
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Vẽ contour (đường đồng mức)
plt.contour(X, Y, Z, levels=20)

# Điểm (3,4)
px, py = 3, 4
plt.plot(px, py, 'ro')  # đánh dấu điểm

# Gradient tại (3,4)
gx, gy = 2*px, 2*py
plt.quiver(px, py, gx, gy, angles='xy', scale_units='xy', scale=1, color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient của f(x,y)=x^2+y^2 tại (3,4)')
plt.axis('equal')
plt.show()
