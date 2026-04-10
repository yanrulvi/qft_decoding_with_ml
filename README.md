# Decoding Quantum Field Theory with Machine Learning

Воспроизведение результатов статьи Grimmer et al. (2023):
"Decoding Quantum Field Theory with Machine Learning"
arXiv:1910.03637v3

## Структура проекта

- `src/field/` — симуляция квантового поля (симплектическая эволюция)
- `src/measurement/` — протокол измерений M0, сжатие данных
- `src/models/` — нейронная сеть
- `src/experiments/` — три физических примера
- `notebooks/` — jupyter-ноутбуки с результатами
- `paper/` — текст курсовой работы

## Установка

```bash
pip install -r requirements.txt
```

## Запуск экспериментов

```bash
python -m src.experiments.boundary
python -m src.experiments.thermometry
python -m src.experiments.fock_vs_coherent
```