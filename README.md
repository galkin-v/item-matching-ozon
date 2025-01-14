# README

## Описание решения для соревнования по матчингу товаров Ozon

В данном проекте мы решаем задачу предсказания схожести товаров на основе их характеристик, используя комбинацию моделей CatBoost и BERT. Основной акцент был сделан на **feature engineering**, который является ключевым этапом в данном решении.

### Описание решения

### 1. **Обзор данных**

- **Тексты описаний товаров**: `text_and_bert_test.parquet`
- **Атрибуты товаров**: `attributes_test.parquet`
- **Идентификаторы товаров для предсказания**: `test.parquet`
- **Векторные представления изображений товаров (ResNet)**: `resnet_test.parquet`

Затем данные объединяются на уровне пар товаров (`variantid1` и `variantid2`), и для каждой пары формируется текстовое описание на основе характеристик и категорий товаров.

### 2. **Feature engineering**
Этот этап был основополагающим для успеха модели. Мы создали множество признаков, включая как традиционные признаки, так и признаки на основе текстов и изображений.

#### 2.1. **Текстовые признаки**
- **BERT embeddings**: Векторные представления текстов получены с помощью модели intfloat/e5-base-v2, предварительно обученной на нашем наборе данных. Мы использовали модель для извлечения эмбеддингов с описания товаров и характеристик.
  
- **Текстовые фичи**:
  - `full_text_var1`, `full_text_var2`: Полные описания товаров.
  - `name_var1`, `name_var2`: Названия товаров.
  - Создан общий текст для каждой пары товаров, объединив названия и характеристики: `text = name_var1 + '\n' + full_text_var1 + '=&=\n' + name_var2 + '\n' + full_text_var2`.

- **Текстовое сходство**:
  - Признак текстового сходства `fuzz_score`, который вычисляется с помощью библиотеки `thefuzz` (метод `token_sort_ratio`). Этот признак оценивает схожесть названий и описаний товаров между двумя товарами.

- **Косинусное сходство BERT эмбеддингов**:
  - Признак текстового сходства на основе BERT эмбеддингов для названий товаров (`bert_64_similarity`).

#### 2.2. **Признаки изображений**
- **ResNet embeddings**: Для каждого товара мы использовали векторные представления изображений, полученные с помощью модели ResNet.
- **Косинусное сходство изображений**: Было рассчитано косинусное сходство эмбеддингов изображений товаров (`main_pic_similarity`), что позволяет оценить визуальное сходство товаров.

#### 2.3. **Атрибуты товаров**
- **Атрибуты товаров (characteristic_attributes_mapping)**: Для каждой пары товаров были использованы атрибуты, такие как размеры, цвет и ISBN. Для некоторых атрибутов использовались специальные правила сравнения:
  - Сравнение **размеров товаров** с учетом корректного формата записи и преобразования значений.
  - Сравнение **цветов** и **ISBN**.

- **Сравнение атрибутов**: Были созданы признаки, указывающие на наличие расхождений по ключевым атрибутам между двумя товарами. Для этого было проверено, соответствуют ли значения ключевых атрибутов, таких как размеры или ISBN, между товарами.

### 3. **Модель**
Для предсказания схожести товаров была использована модель **CatBoostClassifier**, обученная на собранных признаках, включая:
- Текстовые признаки (BERT и fuzz-сходство)
- Признаки изображений (ResNet embeddings и косинусное сходство)
- Атрибуты товаров (сравнение ключевых характеристик)

### 4. **Предсказание и сабмит**
Полученные признаки объединяются в один массив, после чего подаются на вход модели CatBoost. Модель предсказывает вероятность того, что два товара являются похожими. Результаты сохраняются в файл `submission.csv` для отправки.