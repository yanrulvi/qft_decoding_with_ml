import numpy as np
from sklearn.decomposition import PCA


class DataPreprocessor:
    """
    Предобработка данных как в статье (Appendix A):
        1. Центрирование (вычитаем среднее обучающей выборки)
        2. PCA (находим направления независимой вариации)
        3. Отбеливание (приводим дисперсию к 1 по каждому компоненту)

    Важно: fit только на train, transform на train+val+test.
    """

    def __init__(self, n_components: int = None, whiten: bool = True):
        """
        Параметры:
            n_components : число компонент PCA (None = оставить все)
            whiten       : делать ли отбеливание
        """
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
        self.mean_ = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "DataPreprocessor":
        """
        Подгоняет препроцессор на обучающих данных.

        Параметры:
            X_train : shape (n_samples, n_features)
        """
        # Шаг 1: запоминаем среднее обучающей выборки
        self.mean_ = X_train.mean(axis=0)

        # Шаг 2-3: PCA + отбеливание через sklearn
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten
        )
        self.pca.fit(X_train - self.mean_)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Применяет препроцессинг к данным.

        Параметры:
            X : shape (n_samples, n_features)

        Возвращает:
            X_preprocessed : shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor не обучен. Сначала вызови fit().")

        X_centered = X - self.mean_
        return self.pca.transform(X_centered)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Удобный метод: fit + transform на одних данных.
        """
        return self.fit(X_train).transform(X_train)

    def explained_variance_ratio(self) -> np.ndarray:
        """
        Доля объяснённой дисперсии по каждой компоненте.
        Полезно для выбора n_components.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor не обучен.")
        return self.pca.explained_variance_ratio_

    def n_components_for_variance(self, threshold: float = 0.99) -> int:
        """
        Возвращает минимальное число компонент,
        объясняющих threshold долю дисперсии.

        Параметры:
            threshold : например 0.99 = 99% дисперсии
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor не обучен.")
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        return int(np.searchsorted(cumvar, threshold) + 1)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.25,
    seed: int = 42
) -> tuple:
    """
    Делит датасет на train и val.
    Статья использует 75% / 25%.

    Возвращает:
        X_train, X_val, y_train, y_val
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)

    n_val = int(n * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def preprocess_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.25,
    n_components: int = None,
    seed: int = 42
) -> tuple:
    """
    Полный pipeline предобработки:
        1. Делим на train/val
        2. Fit препроцессора на train
        3. Transform train и val

    Возвращает:
        X_train_p, X_val_p  : предобработанные данные
        y_train, y_val      : метки
        preprocessor        : обученный препроцессор (для inference)
    """
    X_train, X_val, y_train, y_val = train_val_split(
        X, y, val_fraction, seed
    )

    preprocessor = DataPreprocessor(
        n_components=n_components,
        whiten=True
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    return X_train_p, X_val_p, y_train, y_val, preprocessor