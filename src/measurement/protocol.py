import numpy as np
from tqdm import tqdm
from src.field.lattice import (
    build_symplectic_form,
    build_hamiltonian_matrix,
    compute_symplectic_evolution,
    evolve_gaussian_state,
    get_probe_state,
    sample_quadratures
)
from src.field.states import combine_probe_field_state


def run_measurement_protocol(
    X_total: np.ndarray,
    sigma_total: np.ndarray,
    omega_d: float,
    omega_field: np.ndarray,
    coupling: np.ndarray,
    t_min: float,
    t_max: float,
    n_times: int,
    n_tom: int,
    rng: np.random.Generator = None
) -> dict:
    """
    Протокол измерений M0 из статьи.

    Для каждого момента времени tm в [t_min, t_max]:
        1. Эволюционируем совместное состояние зонд+поле до tm
        2. Извлекаем редуцированное состояние зонда
        3. Сэмплируем n_tom измерений квадратур q, p, r

    Параметры:
        X_total     : начальный вектор смещений (зонд + поле)
        sigma_total : начальная матрица ковариаций (зонд + поле)
        omega_d     : частота зонда
        omega_field : частоты мод поля, shape (n_field,)
        coupling    : силы связи зонда с модами, shape (n_field,)
        t_min       : начальное время измерений
        t_max       : конечное время измерений
        n_times     : число временных точек
        n_tom       : число измерений в каждой точке (томография)
        rng         : генератор случайных чисел

    Возвращает словарь с сырыми измерениями по всем временам
    """
    if rng is None:
        rng = np.random.default_rng()

    n_field = len(omega_field)
    n_modes = 1 + n_field
    Omega = build_symplectic_form(n_modes)

    # Строим матрицу гамильтониана (без switching function, chi=1)
    F = build_hamiltonian_matrix(
        n_field=n_field,
        omega_d=omega_d,
        omega_field=omega_field,
        coupling=coupling,
        chi=1.0
    )

    times = np.linspace(t_min, t_max, n_times)
    measurements = {"times": times, "q": [], "p": [], "r": []}

    # Эволюционируем инкрементально: S(t_{m+1}) = S(dt) @ S(t_m)
    dt = times[0] if n_times > 0 else t_min
    X_current = X_total.copy()
    sigma_current = sigma_total.copy()

    for i, tm in enumerate(times):
        # Для первой точки эволюционируем от 0 до t_min
        # Для остальных — на шаг dt вперёд
        if i == 0:
            S = compute_symplectic_evolution(F, Omega, tm)
        else:
            dt = times[i] - times[i-1]
            S = compute_symplectic_evolution(F, Omega, dt)

        X_current, sigma_current = evolve_gaussian_state(
            X_current, sigma_current, S
        )

        # Редуцированное состояние зонда
        X_probe, sigma_probe = get_probe_state(X_current, sigma_current)

        # Сэмплируем n_tom измерений
        samples = sample_quadratures(X_probe, sigma_probe, n_tom, rng)

        measurements["q"].append(samples["q"])
        measurements["p"].append(samples["p"])
        measurements["r"].append(samples["r"])

    # Конвертируем в numpy arrays: shape (n_times, n_tom)
    measurements["q"] = np.array(measurements["q"])
    measurements["p"] = np.array(measurements["p"])
    measurements["r"] = np.array(measurements["r"])

    return measurements


def generate_dataset(
    param_sampler,
    state_builder,
    n_samples: int,
    n_times: int,
    n_tom: int,
    t_min: float,
    t_max: float,
    omega_d: float,
    omega_field: np.ndarray,
    coupling_builder,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерирует датасет (X, y) для обучения нейросети.

    Для каждого из n_samples примеров:
        1. Сэмплируем метку y из param_sampler
        2. Строим начальное состояние через state_builder(y)
        3. Запускаем протокол M0
        4. Сжимаем данные в вектор признаков

    Параметры:
        param_sampler   : функция () -> y (метка примера)
        state_builder   : функция (y) -> (X_total, sigma_total, coupling)
        n_samples       : число примеров в датасете
        coupling_builder: функция (y) -> coupling array

    Возвращает:
        X_data : shape (n_samples, 9 * n_times) — сжатые данные
        y_data : shape (n_samples,) — метки
    """
    rng = np.random.default_rng(seed)
    X_data = []
    y_data = []

    for _ in tqdm(range(n_samples), desc="Generating dataset"):
        # Сэмплируем метку
        y = param_sampler(rng)

        # Строим начальное состояние
        X_total, sigma_total = state_builder(y)

        # Строим coupling для данной метки
        coupling = coupling_builder(y)

        # Запускаем протокол
        measurements = run_measurement_protocol(
            X_total=X_total,
            sigma_total=sigma_total,
            omega_d=omega_d,
            omega_field=omega_field,
            coupling=coupling,
            t_min=t_min,
            t_max=t_max,
            n_times=n_times,
            n_tom=n_tom,
            rng=rng
        )

        # Сжимаем данные
        features = compress_measurements(measurements)
        X_data.append(features)
        y_data.append(y)

    return np.array(X_data), np.array(y_data)


def compress_measurements(measurements: dict) -> np.ndarray:
    """
    Сжимает сырые измерения в вектор признаков.

    Для каждого момента времени tm вычисляем:
        - выборочное среднее: q_bar, p_bar, r_bar
        - выборочную дисперсию: s_q, s_p, s_r
        - выборочный 4й центральный момент: s4_q, s4_p, s4_r

    Итого 9 * n_times признаков.
    """
    q = measurements["q"]  # shape (n_times, n_tom)
    p = measurements["p"]
    r = measurements["r"]

    features = []

    for quad in [q, p, r]:
        mean = quad.mean(axis=1)           # shape (n_times,)
        centered = quad - mean[:, None]
        var = (centered**2).mean(axis=1)   # дисперсия
        m4 = (centered**4).mean(axis=1)    # 4й момент
        features.extend([mean, var, m4])

    # Чередуем: q_mean, p_mean, r_mean, q_var, ...
    # Порядок: [q_bar, r_bar, p_bar, s_q, s_r, s_p, s4_q, s4_r, s4_p]
    # (как в статье, уравнение 5)
    q_bar, p_bar, r_bar = features[0], features[3], features[6]
    s_q,   s_p,   s_r   = features[1], features[4], features[7]
    s4_q,  s4_p,  s4_r  = features[2], features[5], features[8]

    vector = np.concatenate([
        q_bar, r_bar, p_bar,
        s_q,   s_r,   s_p,
        s4_q,  s4_r,  s4_p
    ])

    return vector