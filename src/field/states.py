import numpy as np


def vacuum_covariance(n_modes: int) -> np.ndarray:
    """
    Матрица ковариаций вакуумного состояния для n_modes осцилляторов.
    
    Для каждого осциллятора в вакууме:
        <q^2> = 1/2, <p^2> = 1/2, <qp> = 0
    
    Итого sigma = (1/2) * I_{2n}
    """
    return 0.5 * np.eye(2 * n_modes)


def thermal_covariance(omega: np.ndarray, temperature: float) -> np.ndarray:
    """
    Матрица ковариаций теплового состояния поля.
    
    Для каждой моды с частотой omega_n при температуре T:
        n_th = 1 / (exp(omega_n / T) - 1)  -- среднее число фотонов
        <q^2> = <p^2> = n_th + 1/2
    
    Параметры:
        omega       : частоты мод, shape (n_field,)
        temperature : температура T (в единицах hbar=kb=1)
    
    Возвращает матрицу ковариаций размера (2*n_field, 2*n_field)
    """
    n_field = len(omega)
    sigma = np.zeros((2 * n_field, 2 * n_field))

    for n in range(n_field):
        if temperature < 1e-10:
            # T -> 0: вакуумное состояние
            n_th = 0.0
        else:
            exp_arg = omega[n] / temperature
            # Защита от переполнения
            if exp_arg > 500:
                n_th = 0.0
            else:
                n_th = 1.0 / (np.exp(exp_arg) - 1.0)

        variance = n_th + 0.5
        sigma[2*n, 2*n] = variance      # <q_n^2>
        sigma[2*n+1, 2*n+1] = variance  # <p_n^2>

    return sigma


def build_lattice_frequencies(
    n_field: int,
    mass: float,
    lattice_spacing: float
) -> np.ndarray:
    """
    Частоты мод решётки для массивного скалярного поля.
    
    Дисперсионное соотношение (решёточное приближение):
        omega_k = sqrt(m^2 + (2/a)^2 * sin^2(pi*k / (2*N)))
    
    Параметры:
        n_field        : число мод (узлов решётки)
        mass           : масса поля m
        lattice_spacing: шаг решётки a
    
    Возвращает массив частот shape (n_field,)
    """
    k = np.arange(1, n_field + 1)
    arg = np.pi * k / (2 * n_field)
    omega = np.sqrt(mass**2 + (2 / lattice_spacing)**2 * np.sin(arg)**2)
    return omega


def build_coupling_strengths(
    n_field: int,
    sigma_probe: float,
    lattice_spacing: float,
    probe_position: float,
    lambda0: float
) -> np.ndarray:
    """
    Вычисляет силу связи зонда с каждым узлом решётки.
    
    Используется гауссово размытие F(x) = exp(-x^2 / 2sigma^2):
        coupling_n = lambda0 * a * F(x_n - x_probe)
    
    Параметры:
        n_field        : число мод
        sigma_probe    : ширина гауссова размытия зонда
        lattice_spacing: шаг решётки a
        probe_position : позиция зонда x_probe
        lambda0        : константа связи
    
    Возвращает массив coupling shape (n_field,)
    """
    positions = np.arange(1, n_field + 1) * lattice_spacing
    dx = positions - probe_position
    smearing = np.exp(-dx**2 / (2 * sigma_probe**2))
    coupling = lambda0 * lattice_spacing * smearing
    return coupling


def initial_probe_state(omega_d: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Начальное состояние зонда — основное состояние гармонического осциллятора.
    
    X_probe = (0, 0)
    sigma_probe = (1/2) * I_2
    """
    X = np.zeros(2)
    sigma = 0.5 * np.eye(2)
    return X, sigma


def combine_probe_field_state(
    X_probe: np.ndarray,
    sigma_probe: np.ndarray,
    X_field: np.ndarray,
    sigma_field: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Объединяет состояние зонда и поля в одно совместное состояние.
    
    Порядок: (q_d, p_d, q_1, p_1, ..., q_N, p_N)
    
    Поскольку зонд и поле изначально не запутаны,
    матрица ковариаций блочно-диагональна.
    """
    X_total = np.concatenate([X_probe, X_field])

    n_probe = len(X_probe)
    n_field = len(X_field)
    dim = n_probe + n_field

    sigma_total = np.zeros((dim, dim))
    sigma_total[:n_probe, :n_probe] = sigma_probe
    sigma_total[n_probe:, n_probe:] = sigma_field

    return X_total, sigma_total