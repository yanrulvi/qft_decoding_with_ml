import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class QFTDecoder(nn.Module):
    """
    Нейросеть для декодирования свойств квантового поля
    из локальных измерений зонда.

    Архитектура как в статье (Appendix A):
        Input(d) -> Linear(90) -> LeakyReLU
                 -> Linear(30) -> LeakyReLU
                 -> Linear(n_out)

    Для классификации: n_out = n_classes, финальный Softmax
    Для регрессии:     n_out = 1, без финальной активации
    """

    def __init__(
        self,
        input_dim: int,
        task: str = "classification",
        n_classes: int = 2,
        hidden_dims: list = [90, 30],
        negative_slope: float = 0.01
    ):
        """
        Параметры:
            input_dim     : размерность входа (после PCA)
            task          : "classification" или "regression"
            n_classes     : число классов (только для classification)
            hidden_dims   : размеры скрытых слоёв
            negative_slope: параметр LeakyReLU
        """
        super().__init__()

        if task not in ("classification", "regression"):
            raise ValueError("task должен быть 'classification' или 'regression'")

        self.task = task
        self.n_classes = n_classes

        # Строим слои
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            in_dim = hidden_dim

        # Выходной слой
        out_dim = n_classes if task == "classification" else 1
        layers.append(nn.Linear(in_dim, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Trainer:
    """
    Обучение и валидация QFTDecoder.
    """

    def __init__(
        self,
        model: QFTDecoder,
        lr: float = 1e-3,
        l2_lambda: float = 1e-4,
        device: str = None
    ):
        """
        Параметры:
            model     : экземпляр QFTDecoder
            lr        : learning rate
            l2_lambda : коэффициент L2 регуляризации
            device    : "cuda", "cpu" или None (авто)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = model.to(device)

        # Оптимизатор с L2 регуляризацией (weight_decay = l2_lambda)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=l2_lambda
        )

        # Функция потерь
        if model.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def _to_tensors(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> TensorDataset:
        X_t = torch.FloatTensor(X).to(self.device)

        if self.model.task == "classification":
            y_t = torch.LongTensor(y.astype(int)).to(self.device)
        else:
            y_t = torch.FloatTensor(y).to(self.device)

        return TensorDataset(X_t, y_t)

    def train_epoch(self, loader: DataLoader) -> float:
        """Один проход по обучающим данным."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            output = self.model(X_batch)

            if self.model.task == "regression":
                output = output.squeeze(-1)

            loss = self.criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(
        self,
        loader: DataLoader
    ) -> tuple[float, float]:
        """
        Вычисляет loss и accuracy на валидационных данных.

        Для регрессии accuracy = доля примеров в пределах 1% от истины.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                output = self.model(X_batch)

                if self.model.task == "regression":
                    output = output.squeeze(-1)
                    loss = self.criterion(output, y_batch)
                    # Accuracy: предсказание в пределах 1% от истины
                    correct += ((output - y_batch).abs() <= 0.01 * y_batch.abs()).sum().item()
                else:
                    loss = self.criterion(output, y_batch)
                    preds = output.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()

                total_loss += loss.item()
                total += len(y_batch)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True
    ) -> dict:
        """
        Полное обучение модели.

        Возвращает историю обучения.
        """
        train_dataset = self._to_tensors(X_train, y_train)
        val_dataset = self._to_tensors(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

        iterator = range(n_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if verbose:
                iterator.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.3f}"
                })

        return history

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])