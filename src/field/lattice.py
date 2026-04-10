import numpy as np
from scipy.linalg import expm


def build_symplectic_form(n_modes: int) -> np.ndarray:
    """
    Строит симплектическую форму Omega для n_modes осцилляторов.
    Каждый осциллятор представлен парой (q, p).
    
    Omega = diag([J, J, ..., J]) где J = [[0, 1], [-1, 0]]
    """
    J = np.array([[0, 1], [-1, 0]], dtype=float)
    return np.kron(np.eye(n_modes), J)


def build_hamiltonian_matrix(
    n_field: int,
    omega_d: float,
    omega_field: np.ndarray,
    coupling: np.ndarray,
    chi: float = 1.0
) -> np.ndarray:
    """
    Строит матрицу F гамильтониана H = (1/2) X^T F X.
    
    Порядок квадратур: (q_d, p_d, q_1, p_1, q_2, p_2, ...)
    
    Параметры:
        n_field     : число мод поля
        omega_d     : частота зонда
        omega_field : частоты мод поля, shape (n_field,)
        coupling    : сила связи зонда с каждой модой, shape (n_field,)
        chi         : значение switching function в данный момент
    """
    n_total = 1 + n_field  # зонд + моды поля
    dim = 2 * n_total
    F = np.zeros((dim, dim))

    # Гамильтониан зонда: omega_d * (q_d^2 + p_d^2) / 2
    F[0, 0] = omega_d
    F[1, 1] = omega_d

    # Гамильтониан поля: sum_n omega_n * (q_n^2 + p_n^2) / 2
    for n in range(n_field):
        idx = 2 * (n + 1)
        F[idx, idx] = omega_field[n]
        F[idx+1, idx+1] = omega_field[n]

    # Взаимодействие: chi * sum_n lambda_n * q_d * q_n
    # Симметризуем: F[i,j] = F[j,i]
    for n in range(n_field):
        idx = 2 * (n + 1)
        F[0, idx] += chi * coupling[n]
        F[idx, 0] += chi * coupling[n]

    return F


def compute_symplectic_evolution(
    F: np.ndarray,
    Omega: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Вычисляет симплектическую матрицу S(dt) = exp(Omega @ F * dt).
    
    Соответствует унитарной эволюции e^{-iH dt/hbar}.
    """
    return expm(Omega @ F * dt)


def evolve_gaussian_state(
    X: np.ndarray,
    sigma: np.ndarray,
    S: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет симплектическое преобразование к гауссовому состоянию.
    
    X     -> S @ X
    sigma -> S @ sigma @ S.T
    """
    X_new = S @ X
    sigma_new = S @ sigma @ S.T
    return X_new, sigma_new


def get_probe_state(
    X: np.ndarray,
    sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Извлекает редуцированное состояние зонда из полного состояния.
    Зонд — первые два элемента вектора квадратур.
    """
    X_probe = X[:2]
    sigma_probe = sigma[:2, :2]
    return X_probe, sigma_probe


def sample_quadratures(
    X_probe: np.ndarray,
    sigma_probe: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None
) -> dict:
    """
    Сэмплирует измерения квадратур q, p, r = (q+p)/sqrt(2).
    
    Возвращает словарь с массивами измерений.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_q, mu_p = X_probe
    sigma_qq = sigma_probe[0, 0]
    sigma_pp = sigma_probe[1, 1]
    sigma_qp = sigma_probe[0, 1]

    # r = (q + p) / sqrt(2)
    mu_r = (mu_q + mu_p) / np.sqrt(2)
    sigma_rr = (sigma_qq + 2 * sigma_qp + sigma_pp) / 2

    q_samples = rng.normal(mu_q, np.sqrt(sigma_qq), n_samples)
    p_samples = rng.normal(mu_p, np.sqrt(sigma_pp), n_samples)
    r_samples = rng.normal(mu_r, np.sqrt(sigma_rr), n_samples)

    return {"q": q_samples, "p": p_samples, "r": r_samples}