import math

# Definisi Persamaan
def f1(x, y):
    return x**2 + x*y - 10

def f2(x, y):
    return y + 3*x*y**2 - 57

# Fungsi Iterasi (NIMx = 1 → g1A dan g2B)
def g1A(x, y):
    return (10 - x**2) / y

def g2B(x, y):
    try:
        if (3*x + 1) == 0 or (57 - y) < 0:
            return None
        return math.sqrt((57 - y) / (3*x + 1))
    except (ZeroDivisionError, ValueError):
        return None

# Iterasi Titik Tetap (Jacobi)
def iterasi_titik_tetap_jacobi(x0, y0, eps, max_iter=100):
    print("\n=== 1. Iterasi Titik Tetap (Jacobi) ===")
    print(f"{'r':>3} {'x':>12} {'y':>12} {'Δx':>12} {'Δy':>12}")
    print("-"*55)
    for r in range(max_iter):
        x1 = g1A(x0, y0)
        y1 = g2B(x0, y0)
        if x1 is None or y1 is None or math.isnan(x1) or math.isnan(y1):
            print("→ Divergen (akar tidak nyata / pembagi nol)")
            return None
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        print(f"{r:3d} {x1:12.6f} {y1:12.6f} {dx:12.6f} {dy:12.6f}")
        if dx < eps and dy < eps:
            print("→ Konvergen")
            return x1, y1
        x0, y0 = x1, y1
    print("→ Divergen (melebihi iterasi maksimum)")
    return None

# Iterasi Titik Tetap (Seidel)
def iterasi_titik_tetap_seidel(x0, y0, eps, max_iter=100):
    print("\n=== 2. Iterasi Titik Tetap (Seidel) ===")
    print(f"{'r':>3} {'x':>12} {'y':>12} {'Δx':>12} {'Δy':>12}")
    print("-"*55)
    for r in range(max_iter):
        x1 = g1A(x0, y0)
        y1 = g2B(x1, y0)
        if x1 is None or y1 is None or math.isnan(x1) or math.isnan(y1):
            print("→ Divergen (akar tidak nyata / pembagi nol)")
            return None
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        print(f"{r:3d} {x1:12.6f} {y1:12.6f} {dx:12.6f} {dy:12.6f}")
        if dx < eps and dy < eps:
            print("→ Konvergen")
            return x1, y1
        x0, y0 = x1, y1
    print("→ Divergen (melebihi iterasi maksimum)")
    return None

# Newton-Raphson
def newton_raphson(x0, y0, eps, max_iter=100):
    print("\n=== 3. Metode Newton-Raphson ===")
    print(f"{'r':>3} {'x':>12} {'y':>12} {'Δx':>12} {'Δy':>12}")
    print("-"*55)
    for r in range(max_iter):
        df1x = 2*x0 + y0
        df1y = x0
        df2x = 3*y0**2
        df2y = 1 + 6*x0*y0
        det = df1x*df2y - df1y*df2x
        if abs(det) < 1e-12:
            print("→ Divergen (Jacobian nol)")
            return None
        u, v = f1(x0, y0), f2(x0, y0)
        x1 = x0 - (u*df2y - v*df1y) / det
        y1 = y0 - (v*df1x - u*df2x) / det
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        print(f"{r:3d} {x1:12.6f} {y1:12.6f} {dx:12.6f} {dy:12.6f}")
        if dx < eps and dy < eps:
            print("→ Konvergen")
            return x1, y1
        x0, y0 = x1, y1
    print("→ Divergen (melebihi iterasi maksimum)")
    return None

# Metode Secant
def secant_klasik(x0, x1, eps, max_iter=100):
    print("\n=== 4. Metode Secant (Klasik) ===")
    print(f"{'r':>3} {'x':>12} {'f(x)':>12} {'Δx':>12}")
    print("-"*45)

    for r in range(max_iter):
        f_x0 = f1(x0, 0)  
        f_x1 = f1(x1, 0)
        if f_x1 - f_x0 == 0:
            print("→ Divergen (pembagi nol)")
            return None
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        dx = abs(x2 - x1)
        print(f"{r:3d} {x2:12.6f} {f_x1:12.6f} {dx:12.6f}")
        if dx < eps:
            print("→ Konvergen")
            return x2
        x0, x1 = x1, x2
    print("→ Divergen (melebihi iterasi maksimum)")
    return None

# Eksekusi
if __name__ == "__main__":
    x0, y0 = 1.5, 3.5
    eps = 1e-6

    iterasi_titik_tetap_jacobi(x0, y0, eps)
    iterasi_titik_tetap_seidel(x0, y0, eps)
    newton_raphson(x0, y0, eps)

    # Untuk secant, gunakan dua tebakan awal x0 dan x1
    secant_klasik(1.0, 2.0, eps)
