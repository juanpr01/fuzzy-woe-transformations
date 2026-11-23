import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



### Crisp WOE

class Node:
    def __init__(self, is_leaf=True, split_value=None, label=None, cardinality = None):
        self.is_leaf = is_leaf
        self.split_value = split_value
        self.label = label
        self.cardinality = cardinality
        self.left = None
        self.right = None
        
    def node_def(self) -> str:
        if self.is_leaf:
            return f"LEAF | Cardinality = {self.cardinality} | Label = {round(self.label, 5)}"
        else:
            return (f"NODE | Cardinality = {round(self.cardinality, 5)} | "
                    f"Split Value: {round(self.split_value, 5)} | "
                    f"Label: {round(self.label, 5)} | ")

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
    
    def _entropy(self, y):
        if y.size == 0: return 0
        p = np.unique(y, return_counts=True)[1].astype(float)/len(y)
        return -1 * np.sum(p * np.log2(p + 1e-9))

    def _gini_impurity(self, y):
        if y.size == 0: return 0
        p = np.unique(y, return_counts=True)[1].astype(float)/len(y)
        return 1 - np.sum(p**2)

    def _variance(self, y):
        if y.size == 0: return 0
        return np.var(y)
    
    def _information_gain(self, y, mask, func):
        s1 = np.sum(mask)
        s2 = mask.size - s1
        if s1 == 0 or s2 == 0: return 0
        return func(y) - (s1 / float(s1 + s2)) * func(y[mask]) - (s2 / float(s1 + s2)) * func(y[np.logical_not(mask)])

    def _max_information_gain_split(self, y, x, func):
        best_change = None
        split_value = None
        
        for val in np.unique(np.sort(x)):
            mask = x < val
            change = self._information_gain(y, mask, func)
            if best_change is None or change > best_change:
                best_change = change
                split_value = val
                
        return {"split_value": split_value, "gain": best_change}

    def fit(self, X, y, depth=0):
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(is_leaf=True, label=np.mean(y), cardinality=len(y))

        if self.criterion == 'entropy':
            func = self._entropy
        elif self.criterion == 'gini':
            func = self._gini_impurity
        elif self.criterion == 'variance':
            func = self._variance
        else:
            raise ValueError("Invalid criterion specified.")

        split = self._max_information_gain_split(y, X, func)
        split_value = split['split_value']

        if split_value is None:
            return Node(is_leaf=True, label=np.mean(y), cardinality=len(y))

        left_mask = X < split_value
        right_mask = ~left_mask

        if np.all(left_mask) or np.all(right_mask):
            return Node(is_leaf=True, label=np.mean(y), cardinality=len(y))

        left_subtree = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth + 1)

        node = Node(
            is_leaf=False,
            split_value=split_value,
            label=np.mean(y),
            cardinality=len(y)
        )
        node.left = left_subtree
        node.right = right_subtree
        self.tree = node

        return node

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The tree has not been trained yet!")

        def _traverse(node, x):
            if node.is_leaf:
                return node.label
            if x < node.split_value:
                return _traverse(node.left, x)
            else:
                return _traverse(node.right, x)

        return np.array([_traverse(self.tree, x) for x in X])
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node is not None:
            print("  " * depth + node.node_def())
            if not node.is_leaf:
                self.print_tree(node.left, depth + 1)
                self.print_tree(node.right, depth + 1)            
    
    def calculate_woe(self, X, y):
        if self.tree is None:
            raise ValueError("The tree has not been trained yet!")

        splits = []
        def _collect_splits(node):
            if not node.is_leaf:
                splits.append(node.split_value)
                _collect_splits(node.left)
                _collect_splits(node.right)

        _collect_splits(self.tree)
        splits = sorted(splits)

        intervals = [(float('-inf'), splits[0])]
        for i in range(len(splits) - 1):
            intervals.append((splits[i], splits[i + 1]))
        intervals.append((splits[-1], float('inf')))

        woe_values = []
        for interval in intervals:
            mask = (X > interval[0]) & (X <= interval[1])
            good = np.sum(y[mask] == 1)
            bad = np.sum(y[mask] == 0)
            woe = np.log( (good+0.5) / (bad+0.5) )
            woe_values.append((interval, woe))

        return woe_values

def assign_woe(X, woe_values):
    X_woe = np.zeros_like(X, dtype=float)
    for value in woe_values:
        interval, woe = value
        mask = (X > interval[0]) & (X <= interval[1])
        X_woe[mask] = woe
    return X_woe

def woe_df(X_train, y_train, X_test):
    X_train_woe = pd.DataFrame()
    X_test_woe = pd.DataFrame()
    y_train = np.array(y_train)
    for col in X_train.columns:
        print(col)
        tree = DecisionTree(criterion='gini', max_depth=2)
        tree.fit(X_train[col].values, y_train)
        woe_values = tree.calculate_woe(X_train[col].values, y_train)
        
        X_train_woe[f'{col}_woe'] = assign_woe(X_train[col].values, woe_values)
        X_test_woe[f'{col}_woe'] = assign_woe(X_test[col].values, woe_values)
        print('ok')
    return X_train_woe, X_test_woe

# Aplicación de la transformación WoE a un dataframe
def woe_df_na(X_train, y_train, X_test):
    X_train_woe = pd.DataFrame()
    X_test_woe = pd.DataFrame()
    y_train = np.array(y_train)
    for col in X_train.columns:
        print(col)
        
        # Separa los datos no nulos
        non_null_train = X_train[col].notnull()
        non_null_test = X_test[col].notnull()

        # Cálculo de los valores woe para los datos no nulos
        tree = DecisionTree(criterion='gini', max_depth=2)
        tree.fit(X_train.loc[non_null_train, col].values, y_train[non_null_train])
        woe_values = tree.calculate_woe(X_train.loc[non_null_train, col].values, y_train[non_null_train])

        array_woe_values_train = assign_woe(X_train[col][non_null_train], woe_values)
        array_woe_values_test = assign_woe(X_test[col][non_null_test], woe_values)

        # Cálculo de los valores woe para los datos nulos
        good = np.sum(y_train[~non_null_train] == 1)
        bad = np.sum(y_train[~non_null_train] == 0)
        woe_value_missing = np.log( (good+0.5) / (bad+0.5) )

        # Crear el array resultado de train
        result_train = np.zeros_like(non_null_train, dtype=np.float64)

        # Indice para recorrer
        index = 0

        # Llenar el array resultado
        for i in range(len(non_null_train)):
            if non_null_train[i]:
                result_train[i] = array_woe_values_train[index]
                index += 1
            else:
                result_train[i] = woe_value_missing

        # Crear el array resultado de test
        result_test = np.zeros_like(non_null_test, dtype=np.float64)

        # Indice para recorrer
        index = 0

        # Llenar el array resultado
        for i in range(len(non_null_test)):
            if non_null_test[i]:
                result_test[i] = array_woe_values_test[index]
                index += 1
            else:
                result_test[i] = woe_value_missing

        # Creacion de las nuevas variables woe
        X_train_woe[f'{col}_woe'] = result_train
        X_test_woe[f'{col}_woe'] = result_test
        
        print('ok')
        
    return X_train_woe, X_test_woe



### Crisp-Fuzzy WOE

def v(x, alfa, beta):
    if beta != 0:
        if x < alfa - beta / 2:
            return 1
        elif x > alfa + beta / 2:
            return 0
        else:
            return (alfa + beta / 2 - x) / beta
    else:
        if x < alfa:
            return 1
        else:
            return 0

def error_func(y, x, grado_pertenencia_padre_func, alfa, beta, label_left, label_right):
    error = 0
    for i in range(len(x)):
        mu_prima = v(x[i], alfa, beta) * label_left + (1 - v(x[i], alfa, beta)) * label_right
        error_observacion = grado_pertenencia_padre_func(x[i]) * (y[i] - mu_prima) ** 2
        error += error_observacion
    return error

def fibonacci_search_fixed(f, y_fixed, x_fixed, grado_pertenencia_padre_func_fixed, alfa_fixed, label_left_fixed, label_right_fixed, a, b, epsilon=0.001):
    # Generar la secuencia de Fibonacci hasta que F_n >= (b - a) / epsilon
    fib = [0, 1]
    while fib[-1] < (b - a) / epsilon:
        fib.append(fib[-1] + fib[-2])
    
    n = len(fib) - 1
    
    # Inicializar los puntos
    x1 = a + (fib[n-2] / fib[n]) * (b - a)
    x2 = a + (fib[n-1] / fib[n]) * (b - a)
    
    f1 = f(y_fixed, x_fixed, grado_pertenencia_padre_func_fixed, alfa_fixed, x1, label_left_fixed, label_right_fixed)
    f2 = f(y_fixed, x_fixed, grado_pertenencia_padre_func_fixed, alfa_fixed, x2, label_left_fixed, label_right_fixed)
    
    for k in range(n-1, 1, -1):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[k-1] / fib[k]) * (b - a)
            f2 = f(y_fixed, x_fixed, grado_pertenencia_padre_func_fixed, alfa_fixed, x2, label_left_fixed, label_right_fixed)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[k-2] / fib[k]) * (b - a)
            f1 = f(y_fixed, x_fixed, grado_pertenencia_padre_func_fixed, alfa_fixed, x1, label_left_fixed, label_right_fixed)
    
    # Ajustar el punto final si es necesario
    if f1 < f2:
        return (x1, f1)
    else:
        return (x2, f2)

def grado_pertenencia_padre_func(x):
    return 1

def min_error_split_alfa(y, x, alfa, grado_pertenencia_padre_func):
    min_error = None
    beta = 0
    
    mask = x < alfa
    
    label_left = sum(grado_pertenencia_padre_func(xi) * yi for xi, yi, bandera in zip(x, y, mask) if bandera) / sum(grado_pertenencia_padre_func(xi) for xi, bandera in zip(x, mask) if bandera)
    label_right = sum(grado_pertenencia_padre_func(xi) * yi for xi, yi, bandera in zip(x, y, mask) if not bandera) / sum(grado_pertenencia_padre_func(xi) for xi, bandera in zip(x, mask) if not bandera)

    min_beta = 0
    max_beta = 2 * min(max(x) - alfa, alfa - min(x)) 
        
    beta, min_error = fibonacci_search_fixed(
        f=error_func,
        y_fixed=y,
        x_fixed=x,
        grado_pertenencia_padre_func_fixed=grado_pertenencia_padre_func,
        alfa_fixed=alfa,
        label_left_fixed=label_left,
        label_right_fixed=label_right,
        a=min_beta,
        b=max_beta,
        epsilon=0.001
    )
    
    def grado_pertenencia_izquierda_func(xi):
        return v(xi, alfa, beta) * grado_pertenencia_padre_func(xi)

    def grado_pertenencia_derecha_func(xi):
        return (1 - v(xi, alfa, beta)) * grado_pertenencia_padre_func(xi)
    
    return {
        "alfa": alfa,
        "beta": beta,
        "grado_pertenencia_izquierda_func": grado_pertenencia_izquierda_func,
        "grado_pertenencia_derecha_func": grado_pertenencia_derecha_func
    }


def fuzzy_fixed_alfas(x_prueba, y_prueba, grado_pertenencia_padre_func):
    y_prueba = np.array(y_prueba)
    
    tree1 = DecisionTree(criterion='gini', max_depth=1)
    tree1.fit(x_prueba, y_prueba)
    first_split = min_error_split_alfa(
        y_prueba, x_prueba, tree1.tree.split_value, grado_pertenencia_padre_func
    )
    
    left_mask = x_prueba < first_split['alfa']
    right_mask = ~left_mask

    # --- División izquierda ---

    tree2l = DecisionTree(criterion='gini', max_depth=1)
    tree2l.fit(x_prueba[left_mask], y_prueba[left_mask])
    if tree2l.tree is not None:
        second_split_left = min_error_split_alfa(
            y_prueba[left_mask],
            x_prueba[left_mask],
            tree2l.tree.split_value,
            first_split['grado_pertenencia_izquierda_func']
        )
    else:
        # No se puede dividir: copia la función padre y deja el alfa igual
        second_split_left = {
            'alfa': first_split['alfa'],
            'grado_pertenencia_izquierda_func': first_split['grado_pertenencia_izquierda_func'],
            'grado_pertenencia_derecha_func': first_split['grado_pertenencia_izquierda_func'],
        }

    # --- División derecha ---
    tree2r = DecisionTree(criterion='gini', max_depth=1)
    tree2r.fit(x_prueba[right_mask], y_prueba[right_mask])
    if tree2r.tree is not None:
        second_split_right = min_error_split_alfa(
            y_prueba[right_mask],
            x_prueba[right_mask],
            tree2r.tree.split_value,
            first_split['grado_pertenencia_derecha_func']
        )
    else:
        second_split_right = {
            'alfa': first_split['alfa'],
            'grado_pertenencia_izquierda_func': first_split['grado_pertenencia_derecha_func'],
            'grado_pertenencia_derecha_func': first_split['grado_pertenencia_derecha_func'],
        }

    # Con los tres cortes, calculo las WOE normales en esos intervalos:
    splits = [second_split_left['alfa'], first_split['alfa'], second_split_right['alfa']]
    intervals = [
        (float('-inf'), splits[0]),
        (splits[0], splits[1]),
        (splits[1], splits[2]),
        (splits[2], float('inf'))
    ]

    woe_values = []
    for interval in intervals:
        mask = (x_prueba > interval[0]) & (x_prueba <= interval[1])
        good = np.sum(y_prueba[mask] == 1)
        bad = np.sum(y_prueba[mask] == 0)
        woe = np.log((good + 0.5) / (bad + 0.5))
        woe_values.append((interval, woe))

    def woe_function(x):
        woe_pond = (
            second_split_left['grado_pertenencia_izquierda_func'](x) * woe_values[0][1]
            + second_split_left['grado_pertenencia_derecha_func'](x) * woe_values[1][1]
            + second_split_right['grado_pertenencia_izquierda_func'](x) * woe_values[2][1]
            + second_split_right['grado_pertenencia_derecha_func'](x) * woe_values[3][1]
        )
        return woe_pond

    return {
        'woe_values': woe_values,
        'funciones_grado_pertenencia': [
            second_split_left['grado_pertenencia_izquierda_func'],
            second_split_left['grado_pertenencia_derecha_func'],
            second_split_right['grado_pertenencia_izquierda_func'],
            second_split_right['grado_pertenencia_derecha_func']
        ],
        'woe_function': woe_function
    }

def plot_grados_pertenencia_fuzzy_woe_fixed_alfas(fuzzy_woe_fixed_alfas, x_range):
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))

    i = 0
    for funcion in fuzzy_woe_fixed_alfas['funciones_grado_pertenencia']:
        membership_values = [funcion(xi) for xi in x_range]
        woe_value = fuzzy_woe_fixed_alfas['woe_values'][i][1]
        plt.plot(x_range, membership_values, label=f'WOE = {woe_value:.2f}')
        i += 1
        
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Grado de pertenencia', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def woe_and_df(X_train, y_train, X_test):
    X_train_woe_and = pd.DataFrame()
    X_test_woe_and = pd.DataFrame()
    y_train = np.array(y_train)
    for col in X_train.columns:
        print(col)
        fuzzy_woe_fa = fuzzy_fixed_alfas(X_train[col].values, y_train, grado_pertenencia_padre_func)
        
        # Calculate WOE for X_train
        X_train_woe_and[f'{col}_woe_and'] = X_train[col].apply(fuzzy_woe_fa['woe_function'])

        # Calculate WOE for X_test
        X_test_woe_and[f'{col}_woe_and'] = X_test[col].apply(fuzzy_woe_fa['woe_function'])
        print('ok')
    return X_train_woe_and, X_test_woe_and

def woe_and_df_na(X_train, y_train, X_test):
    X_train_woe_and = pd.DataFrame()
    X_test_woe_and = pd.DataFrame()
    y_train = np.array(y_train) 
    for col in X_train.columns:
        print(col)
        
        # Separa los datos no nulos
        non_null_train = X_train[col].notnull()
        non_null_test = X_test[col].notnull()
        
        fuzzy_woe_fa = fuzzy_fixed_alfas(X_train[non_null_train][col].values, y_train[non_null_train], grado_pertenencia_padre_func)
        
        # Calculate WOE for X_train
        array_woe_values_train = X_train[non_null_train][col].apply(fuzzy_woe_fa['woe_function']).values

        # Calculate WOE for X_test
        array_woe_values_test = X_test[non_null_test][col].apply(fuzzy_woe_fa['woe_function']).values
        
        # Cálculo de los valores woe para los datos nulos
        good = np.sum(y_train[~non_null_train] == 1)
        bad = np.sum(y_train[~non_null_train] == 0)
        woe_value_missing = np.log( (good+0.5) / (bad+0.5) )

        # Crear el array resultado de train
        result_train = np.zeros_like(non_null_train, dtype=np.float64)

        # Indice para recorrer
        index = 0

        # Llenar el array resultado
        for i in range(len(non_null_train)):
            if non_null_train[i]:
                result_train[i] = array_woe_values_train[index]
                index += 1
            else:
                result_train[i] = woe_value_missing

        # Crear el array resultado de test
        result_test = np.zeros_like(non_null_test, dtype=np.float64)

        # Indice para recorrer
        index = 0

        # Llenar el array resultado
        for i in range(len(non_null_test)):
            if non_null_test[i]:
                result_test[i] = array_woe_values_test[index]
                index += 1
            else:
                result_test[i] = woe_value_missing

        # Creacion de las nuevas variables woe
        X_train_woe_and[f'{col}_woe_and'] = result_train
        X_test_woe_and[f'{col}_woe_and'] = result_test
        
        print('ok')
    return X_train_woe_and, X_test_woe_and

## Fuzzy WOE

def min_error_split(y, x, grado_pertenencia_padre_func, fuzzy=True):
    min_error = None
    alfa = None
    beta = 0
    
    #Elijo como posibles alfas los x's que tengan un grado de pertenencia positivo al nodo
    lista_posibles_alfas = np.unique(np.sort(x[np.array([grado_pertenencia_padre_func(xi) for xi in x]) > 0]))
        
    for posible_alfa in lista_posibles_alfas[1:]:
        
        mask = x < posible_alfa
        
        label_left = sum(grado_pertenencia_padre_func(xi) * yi for xi, yi, bandera in zip(x, y, mask) if bandera) / sum(grado_pertenencia_padre_func(xi) for xi, bandera in zip(x, mask) if bandera)
        label_right = sum(grado_pertenencia_padre_func(xi) * yi for xi, yi, bandera in zip(x, y, mask) if not bandera) / sum(grado_pertenencia_padre_func(xi) for xi, bandera in zip(x, mask) if not bandera)
        
        error = error_func(y, x, grado_pertenencia_padre_func, posible_alfa, beta, label_left, label_right)

        if min_error is None or error < min_error:
            min_error = error
            alfa = posible_alfa
            label_left_ok = label_left
            label_right_ok = label_right

    if fuzzy:
        min_beta = 0
        max_beta = 2 * min(max(x) - alfa, alfa - min(x)) 
        
        beta, min_error = fibonacci_search_fixed(
            f=error_func,
            y_fixed=y,
            x_fixed=x,
            grado_pertenencia_padre_func_fixed=grado_pertenencia_padre_func,
            alfa_fixed=alfa,
            label_left_fixed=label_left_ok,
            label_right_fixed=label_right_ok,
            a=min_beta,
            b=max_beta,
            epsilon=0.001
        )
    
    def grado_pertenencia_izquierda_func(xi):
        return v(xi, alfa, beta) * grado_pertenencia_padre_func(xi)

    def grado_pertenencia_derecha_func(xi):
        return (1 - v(xi, alfa, beta)) * grado_pertenencia_padre_func(xi)
    
    return {
        "alfa": alfa,
        "beta": beta,
        "grado_pertenencia_izquierda_func": grado_pertenencia_izquierda_func,
        "grado_pertenencia_derecha_func": grado_pertenencia_derecha_func
    }

class FuzzyNode:
    def __init__(self, is_leaf=True, alfa=None, beta=None, label=None, fuzzy_woe=None, grado_pertenencia_func=None, cardinality = None, node_error=None):
        self.is_leaf = is_leaf
        self.alfa = alfa
        self.beta = beta
        self.label = label
        self.fuzzy_woe = fuzzy_woe
        self.grado_pertenencia_func = grado_pertenencia_func
        self.cardinality = cardinality
        self.node_error = node_error 
        self.left = None
        self.right = None
        
    def node_def(self) -> str:
        if self.is_leaf:
            return f"LEAF | Label = {round(self.label, 5)} | Fuzzy WOE = {round(self.fuzzy_woe, 5)}"
        else:
            return (f"NODE | Cardinality = {round(self.cardinality, 5)} | "
                    f"Node Error = {round(self.node_error, 5)} | "
                    f"Alfa: {round(self.alfa, 5)} | "
                    f"Beta: {round(self.beta, 5)} | "
                    f"Label: {round(self.label, 5)} | "
                    f"Fuzzy WOE = {round(self.fuzzy_woe, 5)}")

class FuzzyDecisionTree:
    def __init__(self, max_depth): 
        self.root = None
        self.max_depth = max_depth

    def fit(self, y, x, grado_pertenencia_padre_func, fuzzy):
        self.root = self._grow_tree(y, x, grado_pertenencia_padre_func, fuzzy, depth=0)

    def _grow_tree(self, y, x, grado_pertenencia_padre_func, fuzzy, depth):
        node = FuzzyNode()
        
        node.grado_pertenencia_func = grado_pertenencia_padre_func
        node.label = sum(grado_pertenencia_padre_func(xi) * yi for xi, yi in zip(x, y)) / sum(grado_pertenencia_padre_func(xi) for xi in x)
        
        node.fuzzy_woe = np.log(((sum(grado_pertenencia_padre_func(xi)*yi for xi, yi in zip(x, y)))+0.5) /
        (((sum(grado_pertenencia_padre_func(xi) for xi in x) - sum(grado_pertenencia_padre_func(xi)*yi for xi, yi in zip(x, y)))+0.5)))
        
        node.cardinality = sum(grado_pertenencia_padre_func(xi) for xi in x)
        node.node_error = sum(grado_pertenencia_padre_func(xi) * (yi - node.label)**2 for xi, yi in zip(x, y)) 
        
        # Si solo hay un elemento con grado de pertenencia positivo al nodo, no tiene sentido seguir expandiendo el nodo (y ademas min_error_split da error)
        # De hecho, puede que haya mas de un elemento con grado de pertenencia positivo al nodo, pero que valgan todos lo mismo, entonces tampoco tiene sentido seguir por aqui, porque no vas a poder dividir los datos
        unique_positive_memberships = np.unique(np.sort(x[np.array([grado_pertenencia_padre_func(xi) for xi in x]) > 0]))
        if len(unique_positive_memberships) <= 1:
            node.is_leaf = True
            return node
        
        # Perform the split using min_error_split
        split_result = min_error_split(y, x, grado_pertenencia_padre_func, fuzzy)
        
        node.alfa = split_result['alfa']
        node.beta = split_result['beta']

        # Determine if the node is a leaf or should be split further
        if depth < self.max_depth and self._should_split(node, x):
            node.is_leaf = False

            # Recursive splitting for left and right child nodes
            node.left = self._grow_tree(y, x, split_result['grado_pertenencia_izquierda_func'], fuzzy, depth+1)
            node.right = self._grow_tree(y, x, split_result['grado_pertenencia_derecha_func'], fuzzy, depth+1)
        
        return node

    def _should_split(self, node, x):
        return node.cardinality > 0.01*len(x) and node.node_error > 1
    
    def collect_leaf_nodes(self):
        leaves = []
        self._collect_leaf_nodes(self.root, leaves)
        return leaves

    def _collect_leaf_nodes(self, node, leaves):
        if node.is_leaf:
            leaves.append(node)
        else:
            if node.left is not None:
                self._collect_leaf_nodes(node.left, leaves)
            if node.right is not None:
                self._collect_leaf_nodes(node.right, leaves)
                
    def collect_no_leaf_nodes(self):
        no_leaves = []
        self._collect_no_leaf_nodes(self.root, no_leaves)
        return no_leaves
                
    def _collect_no_leaf_nodes(self, node, no_leaves):
        if not node.is_leaf:
            no_leaves.append(node)
            if node.left is not None:
                self._collect_no_leaf_nodes(node.left, no_leaves)
            if node.right is not None:
                self._collect_no_leaf_nodes(node.right, no_leaves)
                
    def _predict_single(self, x):
        leaves = self.collect_leaf_nodes()
        prediction = sum(node.label*node.grado_pertenencia_func(x) for node in leaves)
        return prediction
    
    def predict(self, x):
        return [self._predict_single(xi) for xi in x]
    
    def _predict_single_fuzzy_woe(self, x):
        leaves = self.collect_leaf_nodes()
        prediction = sum(node.fuzzy_woe*node.grado_pertenencia_func(x) for node in leaves)
        return prediction
    
    def predict_fuzzy_woe(self, x):
        return [self._predict_single_fuzzy_woe(xi) for xi in x]

def print_fuzzy_tree(node, margen=0):
    if node is not None:
        print("-" * margen + node.node_def())
        if not node.is_leaf:
            print_fuzzy_tree(node.left, margen + 1)
            print_fuzzy_tree(node.right, margen + 1)

def plot_membership_functions(tree, x_range):
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    leaf_nodes = tree.collect_leaf_nodes()
        
    for node in leaf_nodes:
        membership_values = [node.grado_pertenencia_func(xi) for xi in x_range]
        plt.plot(x_range, membership_values, label=f'WOE AD = {node.fuzzy_woe:.2f}')
    
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Grado de pertenencia', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def woe_ad_df(X_train, y_train, X_test): #X_train y X_test tienen que tener las mismas columnas
    y_train = np.array(y_train)
    X_train_woe_ad = pd.DataFrame()
    X_test_woe_ad = pd.DataFrame()
    for column in X_train.columns:
        print(column)
        column_data_train = X_train[[column]].values.flatten()
        column_data_test = X_test[[column]].values.flatten()
        fdt = FuzzyDecisionTree(max_depth=2)
        fdt.fit(y_train, column_data_train, grado_pertenencia_padre_func, fuzzy=True)
        X_train_woe_ad[column + '_woe_ad'] = fdt.predict_fuzzy_woe(column_data_train)
        X_test_woe_ad[column + '_woe_ad'] = fdt.predict_fuzzy_woe(column_data_test)
        print('ok')
    return X_train_woe_ad, X_test_woe_ad

def woe_ad_df_na(X_train, y_train, X_test): #X_train y X_test tienen que tener las mismas columnas
    y_train = np.array(y_train)
    X_train_woe_ad = pd.DataFrame()
    X_test_woe_ad = pd.DataFrame()
    for column in X_train.columns:
        print(column)
        
        # Separa los datos no nulos
        non_null_train = X_train[column].notnull()
        non_null_test = X_test[column].notnull()
        
        column_data_train = X_train[[column]][non_null_train].values.flatten()
        column_data_test = X_test[[column]][non_null_test].values.flatten()
        fdt = FuzzyDecisionTree(max_depth=2)
        fdt.fit(y_train[non_null_train], column_data_train, grado_pertenencia_padre_func, fuzzy=True)
        
        array_woe_values_train = fdt.predict_fuzzy_woe(column_data_train)
        array_woe_values_test = fdt.predict_fuzzy_woe(column_data_test)
        
        # Cálculo de los valores woe para los datos nulos
        good = np.sum(y_train[~non_null_train] == 1)
        bad = np.sum(y_train[~non_null_train] == 0)
        woe_value_missing = np.log( (good+0.5) / (bad+0.5) )
        
        # Crear el array resultado de train
        result_train = np.zeros_like(non_null_train, dtype=np.float64)

        # Indice para recorrer
        index = 0

        # Llenar el array resultado
        for i in range(len(non_null_train)):
            if non_null_train[i]:
                result_train[i] = array_woe_values_train[index]
                index += 1
            else:
                result_train[i] = woe_value_missing

        # Crear el array resultado de test
        result_test = np.zeros_like(non_null_test, dtype=np.float64)

        # Indice para recorrer
        index = 0

        # Llenar el array resultado
        for i in range(len(non_null_test)):
            if non_null_test[i]:
                result_test[i] = array_woe_values_test[index]
                index += 1
            else:
                result_test[i] = woe_value_missing
        
        X_train_woe_ad[column + '_woe_ad'] = result_train
        X_test_woe_ad[column + '_woe_ad'] = result_test
        print('ok')
    return X_train_woe_ad, X_test_woe_ad