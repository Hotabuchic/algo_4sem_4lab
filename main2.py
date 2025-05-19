import sys
import time

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QFormLayout, QPlainTextEdit,
    QLineEdit, QPushButton, QComboBox, QFileDialog, QTableView,
    QFrame, QSizePolicy, QHeaderView, QSpacerItem, QMessageBox
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem


class DatasetFillingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Восстановление датасета")
        self.resize(1200, 500)

        self.dataFrame = pd.DataFrame()
        self.originalDataFrame = pd.DataFrame()
        self.dataFrameRemove = pd.DataFrame()
        self.encoders = {}
        self.original_dtypes = {}
        self.date_columns = []
        self.categories = {}
        self.protected_columns = ['Card', 'Passport', 'Name']

        self.initUI()

    def initUI(self):
        mainLayout = QHBoxLayout()
        self.setLayout(mainLayout)

        controlPanel = QWidget()
        controlPanel.setFixedWidth(400)
        controlLayout = QFormLayout()
        controlPanel.setLayout(controlLayout)

        self.removePercentageInput = QLineEdit()
        controlLayout.addRow("Удалить (%)", self.removePercentageInput)

        self.removeButton = QPushButton("Удалить")
        self.removeButton.clicked.connect(self.removeRandomData)
        controlLayout.addRow(self.removeButton)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        controlLayout.addRow(line1)

        self.restoreMethodCombo = QComboBox()
        self.restoreMethodCombo.addItems([
            "Метод подстановки с подбором внутри групп",
            "Метод заполнения значением медианы",
            "Метод восстановления пропущенного значения на основе Zet-алгоритма"
        ])
        controlLayout.addRow("Метод:", self.restoreMethodCombo)

        self.restoreButton = QPushButton("Восстановить")
        self.restoreButton.clicked.connect(self.restoreData)
        controlLayout.addRow(self.restoreButton)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        controlLayout.addRow(line2)

        self.loadButton = QPushButton("Загрузить CSV")
        self.loadButton.clicked.connect(self.loadCSV)
        controlLayout.addRow(self.loadButton)

        self.saveButton = QPushButton("Сохранить CSV")
        self.saveButton.clicked.connect(self.saveCSV)
        controlLayout.addRow(self.saveButton)

        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        controlLayout.addItem(spacer)

        self.statsLabel = QPlainTextEdit("Записей: 0\nПропусков: 0%")
        self.statsLabel.setReadOnly(True)
        self.statsLabel.setStyleSheet("background-color: #f0f0f0;")
        self.statsLabel.setFixedHeight(200)
        controlLayout.addRow(self.statsLabel)

        self.tableView = QTableView()
        self.tableView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        mainLayout.addWidget(controlPanel)
        mainLayout.addWidget(self.tableView)

    def removeRandomData(self):
        percentage = float(self.removePercentageInput.text())
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")


        # Создаем копию DataFrame
        result_df = self.dataFrame.copy()

        # Выбираем только столбцы, которые можно изменять (исключая защищенные)
        editable_columns = [col for col in result_df.columns if col not in self.protected_columns]

        # Вычисляем количество ячеек для замены (только в разрешенных столбцах)
        total_editable_cells = result_df[editable_columns].size
        cells_to_nan = int(total_editable_cells * percentage / 100)

        # Генерируем случайные индексы только для разрешенных столбцов
        flat_indices = np.random.choice(
            total_editable_cells,
            size=cells_to_nan,
            replace=False
        )

        # Преобразуем плоские индексы в позиции (строка, столбец)
        rows, cols = np.unravel_index(flat_indices, (len(result_df), len(editable_columns)))

        # Заменяем значения на NaN (только в выбранных столбцах)
        for row, col in zip(rows, cols):
            col_name = editable_columns[col]  # Получаем имя столбца по индексу
            result_df.at[row, col_name] = np.nan

        self.dataFrame = result_df
        self.dataFrame['Card'] = self.dataFrame['Card'].astype('Int64')
        self.dataFrame['Passport'] = self.dataFrame['Passport'].astype('Int64')
        self.dataFrame['Train'] = self.dataFrame['Train'].astype('Int64')
        self.dataFrameRemove = self.dataFrame.copy()
        self.updateTable()
        self.updateStatistics()

    def updateStatistics(self):
        editable_columns = [col for col in self.dataFrame.columns if col not in self.protected_columns]

        # Вычисляем количество ячеек для замены (только в разрешенных столбцах)
        total_editable_cells = self.dataFrame[editable_columns].size
        missingCells = self.dataFrame.isna().sum().sum()
        missingPercentage = (missingCells / total_editable_cells) * 100 if total_editable_cells > 0 else 0
        self.statsLabel.setPlainText(f"Записей: {len(self.dataFrame)}\nПропусков: {missingPercentage:.2f}%")

    def restoreData(self):
        try:
            if self.dataFrame.empty:
                return

            t_1 = time.perf_counter()
            self.dataFrame = self.convert_to_numeric(self.dataFrame)

            method = self.restoreMethodCombo.currentText()
            if method == "Метод подстановки с подбором внутри групп":
                self.groupBasedImputation(['City to', 'Train', 'Date from', 'Date to', 'Cost'])
            elif method == "Метод заполнения значением медианы":
                self.medianImputation()
            elif method == "Метод восстановления пропущенного значения на основе Zet-алгоритма":
                self.zetAlgorithmImputation()

            errorText = self.calculateRelativeError()
            self.dataFrame = self.convert_back_to_original(self.dataFrame)
            self.updateTable()
            if errorText is not None:
                currentText = self.statsLabel.toPlainText()
                t_2 = time.perf_counter()
                dt = t_2 - t_1
                self.statsLabel.setPlainText(f"{currentText}\n{errorText}\nВремя выполнения: {dt:.4f} сек")
        except Exception as ex:
            print(ex)

    def medianImputation(self):
        df = self.dataFrame.copy()

        # Столбцы для медианной импутации
        median_cols = ['Date from', 'Date to', 'Cost']

        for col in df.columns:
            if df[col].isna().sum() == 0 or col in self.protected_columns:
                continue

            try:
                if col in median_cols:
                    # Медиана для числовых данных (включая Unix-даты)
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    # Мода для остальных типов данных
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else None
                    if mode_val is not None:
                        df[col] = df[col].fillna(mode_val)

            except Exception as e:
                print(f"Ошибка в столбце {col}: {str(e)}")
                continue

        self.dataFrame = df

    def groupBasedImputation(self, group_columns, neighbor_window=1000):
        df = self.dataFrame.copy()

        # Для каждой строки с пропусками
        for idx, row in df[df.isna().any(axis=1)].iterrows():
            current_values = row[group_columns]

            # Определяем окно ближайших строк
            start_idx = max(0, idx - neighbor_window // 2)
            end_idx = min(len(df), idx + neighbor_window // 2)
            window_df = df.iloc[start_idx:end_idx]

            # Вычисляем степень совпадения только с ближайшими строками
            similarity_scores = []
            for candidate_idx, candidate_row in window_df.iterrows():
                if candidate_idx == idx:  # Пропускаем текущую строку
                    continue

                # Считаем количество совпадений в group_columns
                matches = sum(
                    1 for col in group_columns
                    if not pd.isna(current_values[col]) and
                    not pd.isna(candidate_row[col]) and
                    (current_values[col] == candidate_row[col])
                )
                similarity_scores.append((candidate_idx, matches))

            # Сортируем по убыванию совпадений
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # Берем лучшего кандидата (с максимальным числом совпадений)
            if similarity_scores and similarity_scores[0][1] > 0:
                best_match_idx = similarity_scores[0][0]
                best_match = df.loc[best_match_idx]

                # Заполняем ВСЕ пропуски в текущей строке значениями из best_match
                for col in df.columns:
                    if pd.isna(row[col]) and not pd.isna(best_match[col]):
                        df.at[idx, col] = best_match[col]

        self.dataFrame = df
        self.medianImputation()
        self.updateTable()

    def zetAlgorithmImputation(self, z_threshold=1.96):
        median_cols = ['Date from', 'Date to', 'Cost']
        df = self.dataFrame.copy()

        for col in df.columns:
            if not df[col].isna().any() or col in self.protected_columns:
                continue

            fill_value = None
            # Обработка ЧИСЛОВЫХ столбцов
            if col in median_cols:
                col_data = df[col].dropna()
                mean_val = col_data.mean()
                std_val = col_data.std()

                if std_val > 0:
                    z_scores = np.abs((col_data - mean_val) / std_val)
                    non_outliers = col_data[z_scores < z_threshold]
                    fill_value = non_outliers.mean() if not non_outliers.empty else mean_val
                else:
                    fill_value = mean_val

                # Для целых чисел округляем
                if pd.api.types.is_integer_dtype(df[col]):
                    fill_value = int(round(fill_value))

            # Обработка СТРОКОВЫХ/КАТЕГОРИАЛЬНЫХ столбцов
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                fill_value = df[col].mode()[0] if not df[col].mode().empty else None

            # Заполняем пропуски
            if fill_value is not None:
                df[col] = df[col].fillna(fill_value)
            else:
                # Резервное заполнение первым валидным значением
                first_valid = df[col].first_valid_index()
                if first_valid is not None:
                    df[col] = df[col].fillna(df[col][first_valid])

        self.dataFrame = df

    def convert_to_numeric(self, numeric_df):
        df = numeric_df.copy()
        self.original_dtypes = df.dtypes.to_dict()

        for col in df.columns:
            # Сохраняем информацию о датах
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.date_columns.append(col)
                # Конвертируем в Unix timestamp, сохраняя NaN
                df[col] = pd.to_datetime(df[col]).apply(
                    lambda x: x.timestamp() if pd.notna(x) else np.nan
                )

        return df

    def convert_back_to_original(self, numeric_df):
        df = numeric_df.copy()

        for col in df.columns:
            # Восстанавливаем даты
            if col in self.date_columns:
                df[col] = pd.to_datetime(df[col], unit='s')
        return df

    def calculateRelativeError(self):
        current_df = self.dataFrame.copy()
        original_df = self.convert_to_numeric(self.originalDataFrame)

        text = ""
        sum_percntg = 0
        sum_acc = 0

        for col in current_df.columns:
            if col in self.protected_columns:
                continue
            if col in ['Date from', 'Date to', 'Cost']:
                percentage_sum = 0
                for idx in current_df.index:
                    current_val = current_df.at[idx, col]
                    if pd.isna(current_val):
                        current_val = 0
                    original_val = original_df.at[idx, col]

                    if current_val != original_val:
                        percentage_sum += (abs(current_val - original_val) / original_val) * 100

                sum_percntg += percentage_sum
                text += f"Sum avg. dev. {col}: {percentage_sum}%\n"
            else:
                percentage_sum = 0
                all_nan = 0
                for idx in current_df.index:
                    rem_val = self.dataFrameRemove.at[idx, col]
                    if pd.isna(rem_val):
                        all_nan += 1
                        current_val = current_df.at[idx, col]
                        original_val = original_df.at[idx, col]
                        if not pd.isna(current_val) and current_val == original_val:
                            percentage_sum += 1
                sum_acc += (percentage_sum / all_nan) * 100
                text += f"Accuracy {col}: {(percentage_sum / all_nan) * 100}%\n"

        text += f"Full sum average deviation {sum_percntg}%\n"
        text += f"Avg. accuracy {sum_acc / 4}%\n"
        return text

    def updateTable(self):
        if self.dataFrame.empty:
            return

        model = QStandardItemModel()
        model.setColumnCount(len(self.dataFrame.columns))
        model.setRowCount(len(self.dataFrame.index))
        model.setHorizontalHeaderLabels(self.dataFrame.columns.tolist())

        for rowIndex, row in self.dataFrame.iterrows():
            for colIndex, value in enumerate(row):
                item = QStandardItem(str(value))
                model.setItem(rowIndex, colIndex, item)

        self.tableView.setModel(model)

        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        for i in [0, 2, 3]:
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        total = self.dataFrame.size
        missing = self.dataFrame.isna().sum().sum()
        percentMissing = round((missing / total) * 100, 2) if total > 0 else 0
        self.statsLabel.setPlainText(f"Записей: {len(self.dataFrame)}\nПропусков: {percentMissing}%")

    def loadCSV(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
        if path:
            self.dataFrame = pd.read_csv(path, sep=';', parse_dates=['Date from', 'Date to'])
            self.originalDataFrame = self.dataFrame.copy()
            self.updateTable()

    def saveCSV(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV файл", "", "CSV Files (*.csv)")
        if path:
            self.dataFrame.to_csv(path, index=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DatasetFillingApp()
    window.show()
    sys.exit(app.exec_())
