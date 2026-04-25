import numpy as np
import torch
from pathlib import Path

from src.field.lattice import build_symplectic_form
from src.field.states import (
    build_lattice_frequencies,
    build_coupling_strengths,
    initial_probe_state,
    vacuum_covariance,
    combine_probe_field_state
)
from src.measurement.protocol import generate_dataset
from src.measurement.compression import preprocess_dataset
from src.models.network import QFTDecoder, Trainer


# ─── Физические параметры (из статьи, Sec. IV) ───────────────────────────────

BOHR_RADIUS     = 53e-12        # м, размер зонда sigma
UV_CUTOFF_K     = 16 / BOHR_RADIUS
LATTICE_SPACING = np.pi / UV_CUTOFF_K
N_FIELD         = 457           # число узлов решётки
CAVITY_LENGTH   = N_FIELD * LATTICE_SPACING
FIELD_MASS      = 1.0           # эВ (эффективно безмассовое)
OMEGA_D         = 130.0         # эВ, частота зонда
LAMBDA0         = OMEGA_D       # сильная связь

PROBE_POSITION  = LATTICE_SPACING  # зонд у левого края

# ─── Параметры протокола ──────────────────────────────────────────────────────

T_MIN       = 1e-18     # с (аттосекунды)
T_MAX       = 30e-18
N_TIMES     = 30
N_TOM       = 1000      # число измерений на точку (уменьшено для скорости)
N_SAMPLES   = 1500      # число примеров (статья: 15000, уменьшено для теста)
N_EPOCHS    = 50
BATCH_SIZE  = 128

RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_field_frequencies() -> np.ndarray:
    return build_lattice_frequencies(
        n_field=N_FIELD,
        mass=FIELD_MASS,
        lattice_spacing=LATTICE_SPACING
    )


def build_probe_coupling() -> np.ndarray:
    return build_coupling_strengths(
        n_field=N_FIELD,
        sigma_probe=BOHR_RADIUS,
        lattice_spacing=LATTICE_SPACING,
        probe_position=PROBE_POSITION,
        lambda0=LAMBDA0
    )


def build_initial_state_full_bond() -> tuple:
    """
    y=1: Full Bond — граница на последнем узле.
    Стандартная решётка, все связи одинаковы.
    """
    omega_field = build_field_frequencies()

    X_probe, sigma_probe = initial_probe_state(OMEGA_D)
    sigma_field = vacuum_covariance(N_FIELD)
    X_field = np.zeros(2 * N_FIELD)

    X_total, sigma_total = combine_probe_field_state(
        X_probe, sigma_probe, X_field, sigma_field
    )
    return X_total, sigma_total


def build_initial_state_cut_bond() -> tuple:
    """
    y=2: Cut Bond — граница на предпоследнем узле.
    Последний осциллятор отсоединён от решётки.
    """
    # Физически то же начальное состояние —
    # разница только в гамильтониане эволюции.
    # Здесь мы это моделируем через coupling последнего узла = 0
    return build_initial_state_full_bond()


def param_sampler_binary(rng: np.random.Generator) -> int:
    """Сэмплируем метку: 0 (Full Bond) или 1 (Cut Bond)."""
    return int(rng.integers(0, 2))


def state_builder_boundary(y: int) -> tuple:
    """Строим начальное состояние для метки y."""
    # Оба случая имеют одинаковое начальное состояние —
    # разница в гамильтониане (coupling последнего узла)
    return build_initial_state_full_bond()


def coupling_builder_boundary(y: int) -> np.ndarray:
    """
    Строим coupling в зависимости от метки:
        y=0: Full Bond — все узлы связаны одинаково
        y=1: Cut Bond  — последний узел отсоединён
    """
    coupling = build_probe_coupling()

    if y == 1:
        # Обрываем связь последнего узла с остальными
        # через нулевой coupling к нему от зонда
        # (упрощение: полный разрыв через coupling=0 для последних 2 узлов)
        coupling[-1] = 0.0
        coupling[-2] = 0.0

    return coupling


def run_boundary_experiment():
    """
    Главная функция эксперимента по дистанционному зондированию границы.
    Воспроизводит рис. 2 из статьи.
    """
    print("=" * 60)
    print("Эксперимент 1: Дистанционное зондирование границы")
    print("=" * 60)

    omega_field = build_field_frequencies()

    # ── Генерация датасета ────────────────────────────────────────
    print("\n[1/4] Генерация датасета...")
    X_data, y_data = generate_dataset(
        param_sampler=param_sampler_binary,
        state_builder=state_builder_boundary,
        n_samples=N_SAMPLES,
        n_times=N_TIMES,
        n_tom=N_TOM,
        t_min=T_MIN,
        t_max=T_MAX,
        omega_d=OMEGA_D,
        omega_field=omega_field,
        coupling_builder=coupling_builder_boundary,
        seed=42
    )

    print(f"    Датасет: {X_data.shape[0]} примеров, "
          f"{X_data.shape[1]} признаков")

    # ── Предобработка ─────────────────────────────────────────────
    print("\n[2/4] Предобработка (PCA + отбеливание)...")
    X_train, X_val, y_train, y_val, preprocessor = preprocess_dataset(
        X_data, y_data, val_fraction=0.25, seed=42
    )

    n_components = preprocessor.n_components_for_variance(threshold=0.99)
    print(f"    99% дисперсии объясняется {n_components} компонентами")
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}")

    # ── Обучение нейросети ────────────────────────────────────────
    print("\n[3/4] Обучение нейросети...")
    model = QFTDecoder(
        input_dim=X_train.shape[1],
        task="classification",
        n_classes=2
    )
    trainer = Trainer(model, lr=1e-3, l2_lambda=1e-4)

    history = trainer.fit(
        X_train, y_train,
        X_val, y_val,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )

    final_acc = history["val_accuracy"][-1]
    print(f"\n    Финальная val accuracy: {final_acc:.1%}")

    # ── Сохранение результатов ────────────────────────────────────
    print("\n[4/4] Сохранение...")

    torch.save(
        model.state_dict(),
        "results/boundary_model.pt"
    )

    np.save("results/boundary_history.npy", history)

    plot_training_history(history)

    print("\nГотово! Результаты в results/")
    return history, model


def plot_training_history(history: dict):
    """Строит график accuracy во времени обучения."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], label="Train loss")
        ax1.plot(epochs, history["val_loss"], label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss during training")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["val_accuracy"], color="green")
        ax2.axhline(y=0.5, color="gray", linestyle="--",
                    label="Random chance")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation accuracy")
        ax2.set_title("Boundary classification accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "results/figures/boundary_training.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("    График сохранён: results/figures/boundary_training.png")

    except ImportError:
        print("    matplotlib не найден, график пропущен")


if __name__ == "__main__":
    run_boundary_experiment()