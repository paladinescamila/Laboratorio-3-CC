# UNIDAD 4: INTERPOLACIÓN
import time
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt


# Función auxiliar: Ejecuta una función polinómica para una entrada t
def polinomio(n, t, x):
    """
    Entrada: dos enteros n y t, un vector x.
    Salida: f(t) = x1 + x2*t + ... + xn*t^n.
    """

    ft = sum([x[i]*t**i for i in range(n)])
    return ft


# Función auxiliar: Mapea una entrada t_var a un rango de valores en t
def mapear(n, t, t_var):
    """
    Entrada: dos enteros n y t_var, un vector t.
    Salida: índice dentro de t donde comienza el rango en que se encuentra t_var.
    """
    
    for i in range(n - 1):
        if (t_var >= t[i] and t_var <= t[i + 1]):
            return i
    
    return n - 2


# Método de Interpolación Polinomial
def polinomial(t, y):
    """
    Entrada: dos vectores t & y.
    Salida: un vector x de parámetros para un ajuste polinomial por 
            interpolación usando los datos de t (entrada) & y (salida).
    """

    n = len(t)
    A = [[float(i**j) for j in range(n)] for i in t]
    b = [i for i in y]
    x = np.linalg.solve(A, b)

    return x


# Método de Interpolación de Lagrange
def lagrange(t, y):
    """
    Entrada: dos vectores t & y.
    Salida: polinomio interpolante para los datos (ti, yi).
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
    Entrada: dos vectores t & y.
    Salida: polinomio interpolante para los datos (ti, yi).
    """

    n = len(t)
    A = [[1.0 for _ in range(n)] for _ in range(n)]
    b = [i for i in y]

    for i in range(n):
        for j in range(n):
            for k in range(j):
                A[i][j] *= (t[i] - t[k])
    
    x = np.linalg.solve(A, b)

    pol, t_sym = 0, sym.Symbol("t")
    for i in range(n):
        base = np.prod([t_sym - t[k] for k in range(i)])
        pol += x[i] * base

    return pol


# Método de Interpolación Lineal a Trozos
def lineal_a_trozos(t, y):
    """
    Entrada: dos vectores t & y.
    Salida: conjunto de parámetros x de un ajuste polinomial de cada 
            par de puntos consecutivos (ti, yi).
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
    Entrada: cuatro vectores te, ye, yv, yv que contienen los datos de 
            entrenamiento y validación, un entero "metodo" que define el 
            método por el cual se va a calcular y un booleano "mostrar" que
            determina si se grafica e imprime los resultados obtenidos.
    Salida: error absoluto promedio, desviación estándar del error, y tiempo 
            de cómputo para el método seleccionado.
    """

    t_sym = sym.Symbol("t")
    ne, nv = len(te), len(tv)
    min_t, max_t = 1, ne + nv
    t_funcion = np.linspace(min_t, max_t, 1000)
    
    if (metodo == 1):

        if (mostrar): plt.title("Interpolación Polinomial")
        inicio = time.time()
        x = polinomial(te, ye)
        tiempo = time.time() - inicio

        pol = polinomio(ne, t_sym, x)
        errores = [np.abs(polinomio(ne, tv[i], x) - yv[i]) for i in range(nv)]
        y_funcion = [polinomio(ne, i, x) for i in t_funcion]

    elif (metodo == 2):

        if (mostrar): plt.title("Interpolación de Lagrange")
        inicio = time.time()
        pol = lagrange(te, ye)
        tiempo = time.time() - inicio

        f = sym.lambdify(t_sym, pol)
        errores = [np.abs(f(tv[i]) - yv[i]) for i in range(nv)]
        y_funcion = [f(i) for i in t_funcion]
    
    elif (metodo == 3):

        if (mostrar): plt.title("Interpolación de Newton")
        inicio = time.time()
        pol = newton(te, ye)
        tiempo = time.time() - inicio

        f = sym.lambdify(t_sym, pol)
        errores = [np.abs(f(tv[i]) - yv[i]) for i in range(nv)]
        y_funcion = [f(i) for i in t_funcion]
    
    else:

        if (mostrar): plt.title("Interpolación Lineal a Trozos")
        inicio = time.time()
        soluciones = lineal_a_trozos(te, ye)
        tiempo = time.time() - inicio

        errores = []
        for i in range(nv):
            xi = soluciones[mapear(ne, te, tv[i])]
            errores += [np.abs(polinomio(2, tv[i], xi) - yv[i])]
        
        y_funcion = []
        for i in t_funcion:
            xi = soluciones[mapear(ne, te, i)]
            y_funcion += [polinomio(2, i, xi)]

    error_promedio = np.mean(errores)
    error_desviacion = np.std(errores)

    if (mostrar):
        
        plt.plot(te[0], ye[0], 'ro', markersize=4, color="blue", label="Entrenamiento")
        plt.plot(tv[0], yv[0], 'ro', markersize=4, color="red", label="Validación")
        plt.plot(t_funcion, y_funcion, color="black", label="Polinomio")

        for i in range(1, ne): plt.plot(te[i], ye[i], 'ro', markersize=4, color="blue")
        for i in range(1, nv): plt.plot(tv[i], yv[i], 'ro', markersize=4, color="red")

        plt.legend()
        plt.xlabel('t')
        plt.ylabel('y')
        plt.grid()
        plt.show()

        if (metodo != 4): print("f(t, x) = {}".format(sym.expand(pol)))
        print("Tiempo = {:.5f}s".format(tiempo))
        print("Error (promedio) = {:.2f}".format(error_promedio))
        print("Error (desviación estándar) = {:.2f}\n".format(error_desviacion))

    return error_promedio, error_desviacion, tiempo


# Procesamiento de los datos (adaptado a los ejemplos)
def procesar(url, N, porcentaje_e):
    """
    Entrada: url del conjunto de datos, un entero N que corresponde a la 
            cantidad total de datos a extraer y un entero "porcentaje_e" que 
            indica el porcentaje de datos de entrenamiento.
    Salida: datos de entrenamiento te (entradas) y ye (salidas), y
            de validación tv (entradas) y yv (salidas).
    """

    datos = pd.read_csv(url, header=None)
    y = datos[1].tolist()[:N]
    ne = int(N * porcentaje_e / 100)
    
    te = [int(i) for i in np.linspace(1, N, ne)]
    ye = [y[i - 1] for i in te]
    tv = list(set([i + 1 for i in range(N)]) - set(te))
    yv = [y[i - 1] for i in tv]

    return te, ye, tv, yv


# EJEMPLOS DE PRUEBA (También se encuentran en el informe)
def main():
    
    print("EJEMPLO 1")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-3-CC/main/oro.csv"
    # url = "oro.csv" # URL alternativa para ejecutar de manera local
    te, ye, tv, yv = procesar(url, 100, 5)
    resolver(te, ye, tv, yv, 1, True)
    resolver(te, ye, tv, yv, 2, True)
    resolver(te, ye, tv, yv, 3, True)
    te, ye, tv, yv = procesar(url, 300, 50)
    resolver(te, ye, tv, yv, 4, True)

    print("EJEMPLO 2")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-3-CC/main/python.csv"
    # # url = "python.csv" # URL alternativa para ejecutar de manera local
    te, ye, tv, yv = procesar(url, 50, 13)
    resolver(te, ye, tv, yv, 1, True)
    resolver(te, ye, tv, yv, 2, True)
    resolver(te, ye, tv, yv, 3, True)
    te, ye, tv, yv = procesar(url, 100, 50)
    resolver(te, ye, tv, yv, 4, True)


main()


# ----------------------------------------------------------------------
# ANÁLISIS DE COMPLEJIDAD Y EXACTITUD DE LOS MÉTODOS

# Imprimir las tablas
def imprimir(titulo, porcentajes, valores, decimal):

    print("-------------------------------------------------------------------")
    print("                           {}                            ".format(titulo))
    print("-------------------------------------------------------------------")
    print("%\tPolinomial\tLagrange\tNewton\t\tA trozos")
    print("-------------------------------------------------------------------")

    p = len(porcentajes)
    for i in range(p):
        e_polinomial = valores[i][0]
        e_lagrange = valores[i][1]
        e_newton = valores[i][2]
        e_trozos = valores[i][3]
        if (decimal):
            print("{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(porcentajes[i], e_polinomial, e_lagrange, e_newton, e_trozos))
        else:
            print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(porcentajes[i], e_polinomial, e_lagrange, e_newton, e_trozos))
            
    print("-------------------------------------------------------------------")


# Comparación de los métodos
def estadisticas(url, N_pol, N_trozos):

    metodos = ['Polinomial', 'Lagrange', 'Newton', 'A trozos']
    colores = ['red', 'green', 'orange', 'blue']
    lineas = ['-', '--', '-.', ':']

    porcentajes = [i for i in range(2, 23, 2)]
    errores, desviaciones, tiempos = [], [], []

    for i in porcentajes:
        
        te, ye, tv, yv = procesar(url, N_pol, i)
        e_prom, e_desv, tiempo = [], [], []

        for j in range(3):
            ep, ed, t = resolver(te, ye, tv, yv, j + 1, False)
            e_prom += [ep]
            e_desv += [ed]
            tiempo += [t]
        
        te, ye, tv, yv = procesar(url, N_trozos, i)
        ep, ed, t = resolver(te, ye, tv, yv, 4, False)

        errores += [e_prom + [ep]]
        desviaciones += [e_desv + [ed]]
        tiempos += [tiempo + [t]]

    imprimir("Error (Promedio)", porcentajes, errores, False)
    imprimir("Error (Desviación)", porcentajes, desviaciones, False)

    for i in range(4):
        errores_metodo = [errores[j][i] for j in range(len(porcentajes))]
        plt.plot(porcentajes, errores_metodo, lineas[i], color=colores[i], label=metodos[i])

    plt.title("Error (Promedio)")
    plt.legend()
    plt.xlabel("Porcentaje de entrenamiento")
    plt.ylabel("Error")
    plt.grid()
    plt.show()

    imprimir("Tiempo de ejecución", porcentajes, tiempos, True)

    for i in range(4):
        tiempos_metodo = [tiempos[j][i] for j in range(len(porcentajes))]
        plt.plot(porcentajes, tiempos_metodo, marker="o", color=colores[i], label=metodos[i])
        
    plt.title("Tiempo de ejecución")
    plt.legend()
    plt.xlabel("Porcentaje de entrenamiento")
    plt.ylabel("Tiempo")
    plt.grid()
    plt.show()


# Análisis de la Interpolación Lineal a Trozos
def estadisticas_a_trozos(url, N):

    porcentajes = [i for i in range(2, 100)]
    errores, desviaciones, tiempos = [], [], []

    for i in porcentajes:

        te, ye, tv, yv = procesar(url, N, i)
        ep, ed, t = resolver(te, ye, tv, yv, 4, False)

        errores += [ep]
        desviaciones += [ed]
        tiempos += [t]

    print("Error en la Interpolación Lineal a Trozos")
    plt.title("Error en la Interpolación Lineal a Trozos")
    plt.plot(porcentajes, errores, marker="o", color="blue")
    plt.xlabel("Porcentaje de entrenamiento")
    plt.ylabel("Error")
    plt.grid()
    plt.show()


print("EJEMPLO 1")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-3-CC/main/oro.csv"
estadisticas(url, 100, 200)
estadisticas_a_trozos(url, 200)

print("EJEMPLO 2")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-3-CC/main/python.csv"
estadisticas(url, 50, 100)
estadisticas_a_trozos(url, 100)