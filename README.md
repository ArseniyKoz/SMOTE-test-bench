# SMOTE-test-brench
Тестовый стенд для сравнения oversampling-алгоритмов семейства SMOTE на задачах бинарной классификации с логированием экспериментов в ClearML.

## Overview
Проект автоматизирует benchmark oversampling-методов на фиксированном пуле классификаторов и метрик. Основная цель: сравнить качество модели на исходных данных и после балансировки train-части.

Ключевые возможности:
- загрузка датасетов из ClearML Dataset Registry;
- запуск серий экспериментов по набору методов;
- единый расчет метрик качества для baseline и oversampling;
- логирование таблиц, графиков и артефактов в ClearML.

## Status / Limitations
Текущее состояние проекта:
- поддерживается только бинарная классификация;
- основной источник oversampling-методов: библиотека `smote-variants` через `configs/methods.yaml`;
- кастомные реализации из `src/methods/classic/*` считаются неактуальными и не входят в текущий benchmark-пайплайн.

## Requirements
- Python 3.10+ (рекомендуется 3.11);
- доступ к ClearML Server;
- настроенные credentials ClearML для текущего окружения.

Зависимости указаны в [requirements.txt](./requirements.txt).

## Installation
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Установка пакетов:
```bash
pip install -r requirements.txt
```

Инструменты разработки (тесты/линт/формат):
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Инициализация ClearML (один раз на окружение):
```bash
clearml-init
```

## Development Checks
Базовые проверки перед PR:

```bash
python -m compileall src experiments configs main.py
ruff check tests main.py configs/config_loader.py
black --check tests main.py configs/config_loader.py
pytest -q
```

Эти же шаги выполняются в CI (`.github/workflows/ci.yml`).

## Data Preparation
Конфиг датасетов: [configs/data/datasets.yaml](./configs/data/datasets.yaml).

Для каждого датасета задается:
- `data_id`: ID набора данных в ClearML;
- `prep_data_id`: резервное поле (если используется отдельно подготовленная версия).

Ожидается, что CSV-файл внутри ClearML Dataset:
- имеет имя `<dataset_name>.csv`;
- содержит target в последнем столбце.

## Experiment Configuration
Основные конфиги:
- [configs/experiment/per_dataset.yaml](./configs/experiment/per_dataset.yaml): один датасет, несколько методов;
- [configs/experiment/per_method.yaml](./configs/experiment/per_method.yaml): один метод, несколько датасетов;
- [configs/methods.yaml](./configs/methods.yaml): доступные oversampling-методы (`smote-variants`);
- [configs/config_loader.py](./configs/config_loader.py): загрузка YAML-конфигов.

### `per_dataset.yaml`
Ключи:
- `dataset`: имя датасета (должно совпадать с ключом в `datasets.yaml`);
- `methods`: список методов, которые будут запущены;
- `experiment_config`: параметры CV, random state, список метрик, классификаторы.

### `methods.yaml`
Ключи:
- `<MethodName>`: логическое имя метода;
- `method`: строка конструктора алгоритма из `smote_variants`.

## Run (Target Scenario)
Точка входа: [main.py](./main.py).

Запуск:
```bash
python main.py
```

Целевой сценарий:
- загружается конфиг эксперимента;
- для каждого метода создается отдельная задача в ClearML;
- выполняются CV и финальная оценка на holdout;
- сохраняются таблицы метрик и артефакты результатов.

## Metrics and Evaluation
В проекте используются метрики качества для имбалансных задач, в том числе:
- `balanced_accuracy`;
- `f1_weighted`;
- `g_mean`;
- `roc_auc_weighted`;
- `precision_weighted`;
- `recall_weighted`.

Сравнение проводится в формате:
- `Original`: обучение на исходной train-выборке;
- `SMOTE`: обучение на oversampled train-выборке;
- `Improvement`: разница метрик между двумя режимами.

## Project Structure
```text
SMOTE-test-brench/
├─ configs/
│  ├─ data/
│  │  └─ datasets.yaml
│  ├─ experiment/
│  │  ├─ per_dataset.yaml
│  │  └─ per_method.yaml
│  ├─ methods.yaml
│  └─ config_loader.py
├─ data/
│  └─ dataset_to_clearML.py
├─ experiments/
│  ├─ experiment_runner.py
│  └─ results_printer.py
├─ src/
│  ├─ evaluation/
│  │  └─ basic_evaluator.py
│  ├─ methods/
│  │  ├─ base.py
│  │  └─ classic/
│  └─ utils/
│     ├─ data_loader.py
│     ├─ preprocessing.py
│     └─ visualise.py
├─ main.py
└─ requirements.txt
```

## Troubleshooting
1. `Dataset ID not found` или загрузка датасета не выполняется.
- Проверьте `data_id` в `configs/data/datasets.yaml`.
- Убедитесь, что пользователь имеет доступ к датасету в ClearML.

2. Ошибка чтения CSV (`file not found`).
- Убедитесь, что имя файла внутри датасета совпадает с `dataset` из конфига: `<dataset>.csv`.

3. Ошибки аутентификации ClearML.
- Повторно выполните `clearml-init`.
- Проверьте актуальность API key/secret и URL сервера.

4. Нестабильное качество на разных запусках.
- Проверьте `random_state` в конфиге эксперимента.
- Убедитесь, что сравниваются одинаковые классификаторы и метрики.

## Roadmap
- поддержка multiclass-задач;
- расширение набора встроенных сценариев benchmark;
- повышение воспроизводимости и покрытие автоматическими тестами.
