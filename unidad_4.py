# UNIDAD 4: INTERPOLACIÓN
import time
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt


# Función auxiliar: Método de Sustitución Sucesiva hacia atrás
def sucesiva_hacia_atras(A, b):
    """
    Entrada: una matriz triangular superior A y un vector b.
    Salida: un vector x tal que Ax = b.
    """
    if (np.linalg.det(A) == 0):
        print("A es una matriz singular, el sistema no tiene solución.")
        return []
    else:
        n = len(A) - 1
        x = [None for _ in range(n)] + [b[n] / A[n][n]]
        for i in range(n, -1, -1):
            sumatoria = 0
            for j in range(i+1, n+1):
                sumatoria += A[i][j] * x[j]
            x[i] = round((b[i] - sumatoria) / A[i][i], 5)

        return x


# Función auxiliar: Método de Sustitución Sucesiva hacia adelante
def sucesiva_hacia_adelante(A, b):
    """
    Entrada: una matriz triangular inferior A y un vector b.
    Salida: un vector x tal que Ax = b.
    """
    if (np.linalg.det(A) == 0):
        print("A es una matriz singular, el sistema no tiene solución.")
        return []
    else:
        n = len(A) - 1
        x = [b[0] / A[0][0]] + [None for _ in range(n)]
        for i in range(1, n+1):
            sumatoria = 0
            for j in range(i):
                sumatoria += A[i][j] * x[j]
            x[i] = round((b[i] - sumatoria) / A[i][i], 5)

        return x


# Función auxiliar: Construye la matriz de eliminación para una columna
def matriz_de_eliminacion(A, k, g):
    """
    Entrada: una matriz cuadrada A, un entero k y un booleano g.
    Salida: matriz de eliminación de Gauss (si g es verdadero) o matriz de 
            eliminación de Gauss-Jordan (si g es falso) para la columna Ak.
    """
    n = len(A)
    M = np.identity(n)
    for i in range(k+1, n):
        M[i][k] = (-1) * A[i][k] / A[k][k]
    if (not g):
        for i in range(k):
            M[i][k] = (-1) * A[i][k] / A[k][k]
    
    return M


# Función auxiliar: Permuta una matriz y un vector dados con respecto a una fila de A
def permutar(A, b, k):
    """
    Entrada: una matriz A, un vector b y un entero k.
    Salida: una matriz A y un vector b permutados con respecto a k,
            además de un booleano que determina si el nuevo valor
            del pivote es cero.
    """
    n = len(A)
    i = k + 1
    while (i != n and A[k][k] == 0):
        P = np.identity(n)
        P[k], P[i], P[k][i], P[i][k] = 0, 0, 1, 1
        A = np.matmul(P, A)
        b = np.matmul(P, b)
        i += 1
    cero = A[k][k] == 0

    return A, b, cero


# Función auxiliar: Método de Eliminación de Gauss
def gauss(A, b):
    """
    Entrada: una matriz cuadrada A y un vector b.
    Salida: un vector x tal que Ax = b.
    """
    if (np.linalg.det(A) == 0):
        print("A es una matriz singular, el sistema no tiene solución.")
        return []
    else:
        n = len(A)
        for k in range(n - 1):
            if (A[k][k] == 0):
                A, b, cero = permutar(A, b, k)
                if (cero):
                    print("El sistema no tiene solución.")
                    return []
            M = matriz_de_eliminacion(A, k, 1)
            A = np.matmul(M, A)
            b = np.matmul(M, b)
        x = sucesiva_hacia_atras(A, b)

        return x


# Función auxiliar: Ejecuta una función polinómica para una entrada t
def polinomio(n, t, x):
    """
    Entrada: dos enteros n y t, un vector x.
    Salida: f(t) = x1 + x2*t + ... + xn*t^n.
    """

    ft = sum([x[i]*t**i for i in range(n)])
    return ft


# Función auxiliar: Mapeo de una entrada a un rango de valores
def mapear(n, t, t_var):
    """
    """
    
    for i in range(n - 1):
        if (t_var >= t[i] and t_var <= t[i+1]):
            return i
    
    return n - 2


# Método de Interpolación Polinomial
def polinomial(t, y):
    """
    Entrada: un vector t y un vector y.
    Salida: un vector x de parámetros para un ajuste polinomial
            por interpolación usando los datos de t (entrada) 
            & y (salida).
    """

    n = len(t)
    A = [[float (i**j) for j in range(n)] for i in t]
    b = [i for i in y]
    x = gauss(A, b)

    return x


# Método de Interpolación de Lagrange
def lagrange(t, y):
    """
    """

    n, pol = len(t), 0
    t_sym = sym.Symbol("t")

    for j in range(n):
        numerador, denominador = 1, 1
        for k in range(n):
            if (k != j):
                numerador *= t_sym - t[k]
                denominador *= t[j] - t[k]
        pol += (numerador / denominador) * y[j]

    return pol


# Método de Interpolación de Newton
def newton(t, y):
    """
    """

    n = len(t)
    A = [[1.0 for _ in range(n)] for _ in range(n)]
    b = [i for i in y]

    for i in range(n):
        for j in range(n):
            for k in range(j):
                A[i][j] *= (t[i] - t[k])
    
    x = sucesiva_hacia_adelante(A, b)

    pol, t_sym = 0, sym.Symbol("t")
    for i in range(n):
        base = np.prod([t_sym - t[k] for k in range(i)])
        pol += x[i] * base

    return pol


# Método de Interpolación Lineal a Trozos
def lineal_a_trozos(t, y):
    """
    """

    n = len(t)
    soluciones = []
    
    for i in range(n - 1):
        a, b, c = 1, t[i], y[i]
        d, e, f = 1, t[i + 1], y[i + 1]
        det = a * e - b * d
        x1 = (e * c - b * f) / det
        x2 = (a * f - d * c) / det
        soluciones += [[x1, x2]]
    
    return soluciones


# Muestra los resultados, tiempo y error absoluto promedio
def resolver(te, ye, tv, yv, metodo, mostrar):
    """
    """

    n = len(te)
    t_sym = sym.Symbol("t")
    min_t, max_t = min(min(te), min(tv)), max(max(te), max(tv))
    t_funcion = np.linspace(min_t, max_t, 1000)
    
    if (metodo == 1):

        if (mostrar): plt.title("Interpolación Polinomial")
        inicio = time.time()
        x = polinomial(te, ye)
        tiempo = time.time() - inicio

        pol = polinomio(n, t_sym, x)
        errores = [np.abs(polinomio(n, tv[i], x) - yv[i]) for i in range(n)]
        y_funcion = [polinomio(n, i, x) for i in t_funcion]

    elif (metodo == 2):

        if (mostrar): plt.title("Interpolación de Lagrange")
        inicio = time.time()
        pol = lagrange(te, ye)
        tiempo = time.time() - inicio

        f = sym.lambdify(t_sym, pol)
        errores = [np.abs(f(tv[i]) - yv[i]) for i in range(n)]
        y_funcion = [f(i) for i in t_funcion]
    
    elif (metodo == 3):

        if (mostrar): plt.title("Interpolación de Newton")
        inicio = time.time()
        pol = newton(te, ye)
        tiempo = time.time() - inicio

        f = sym.lambdify(t_sym, pol)
        errores = [np.abs(f(tv[i]) - yv[i]) for i in range(n)]
        y_funcion = [f(i) for i in t_funcion]
    
    else:

        if (mostrar): plt.title("Interpolación Lineal a Trozos")
        inicio = time.time()
        soluciones = lineal_a_trozos(te, ye)
        tiempo = time.time() - inicio

        errores = []
        for i in range(n):
            xi = soluciones[mapear(n, te, tv[i])]
            errores += [np.abs(polinomio(2, tv[i], xi) - yv[i])]
        
        y_funcion = []
        for i in range(1000):
            xi = soluciones[mapear(n, te, t_funcion[i])]
            y_funcion += [polinomio(2, t_funcion[i], xi)]

    error_promedio = np.mean(errores)
    error_desviacion = np.std(errores)

    if (mostrar):

        plt.plot(t_funcion, y_funcion, color="black")
        
        for i in range(n): plt.plot(te[i], ye[i], marker="o", markersize=4, color="blue")
        for i in range(n): plt.plot(tv[i], yv[i], marker="o", markersize=4, color="red")

        plt.xlabel('t')
        plt.ylabel('y')
        plt.grid()
        plt.show()

        if (metodo != 4): print("f(t, x) = {0}".format(sym.expand(pol)))
        print("Tiempo = {0:.5f}s".format(tiempo))
        print("Error (promedio) = {0:.2f}".format(error_promedio))
        print("Error (desviación estándar) = {0:.2f}\n".format(error_desviacion))

    return error_promedio, error_desviacion, tiempo


# Procesamiento de los datos (adaptado a los ejemplos)
def procesar(url, N):
    """
    Entrada: url del conjunto de datos.
    Salida: datos de entrenamiento te (entradas) y ye (salidas), y
            de validación tv (entradas) y yv (salidas).
    """

    datos = pd.read_csv(url, header=None)
    y = datos[1].tolist()[-N:]

    te = [i for i in range(1, N, 2)]
    ye = [y[i] for i in range(0, N, 2)]
    tv = [i for i in range(2, N+1, 2)]
    yv = [y[i] for i in range(1, N, 2)]

    return te, ye, tv, yv


# EJEMPLOS DE PRUEBA (También se encuentran en el informe)
def main():
    
    print("EJEMPLO 1")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
    # url = "muertos.csv" # URL alternativa para ejecutar de manera local
    te, ye, tv, yv = procesar(url, 10)
    resolver(te, ye, tv, yv, 1, True)
    resolver(te, ye, tv, yv, 2, True)
    resolver(te, ye, tv, yv, 3, True)
    resolver(te, ye, tv, yv, 4, True)


main()


# ----------------------------------------------------------------------
# ANÁLISIS DE COMPLEJIDAD Y EXACTITUD DE LOS MÉTODOS

# Comparación de los métodos
def estadisticas(url, N):
    metodos = ['Polinómica', 'Lagrange', 'Newton', 'A trozos']
    te, ye, tv, yv = procesar(url, N)
    errores, tiempos = [], []
    for i in range(4):
        ep, ed, t = resolver(te, ye, tv, yv, i+1, False)
        errores += [ep]
        tiempos += [t]

    pd.DataFrame({'Error': errores}, index=metodos).plot(kind='bar')
    pd.DataFrame({'Tiempo': tiempos}, index=metodos).plot(kind='bar')


url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
estadisticas(url, 10)