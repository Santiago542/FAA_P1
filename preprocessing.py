from typing import Any

import numpy as np
import pandas as pd


class Dataset:
    """Representación de un dataset cargado desde un fichero CSV."""

    def __init__(self, file_path: str):
        """
        Inicializa y carga el dataset desde un fichero CSV.

        Args:
            file_path (str): Ruta al fichero CSV.
        """
        self.file_path: str = file_path
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.feature_names: list[str] = []
        self.nominal_attributes: list[bool] = []
        self.dictionaries: list[dict[Any, int]] = []
        self.column_dictionaries: dict[str, dict[Any, int]] = {}
        self.data: np.ndarray = np.array([])

        self._load_and_preprocess()

    def _load_and_preprocess(self) -> None:
        """Carga el CSV, detecta tipos de atributos, crea mapeos lexicográficos y codifica los datos."""
        self.raw_df = pd.read_csv(self.file_path)
        self.feature_names = list(self.raw_df.columns)

        self.nominal_attributes = []
        self.dictionaries = []
        self.column_dictionaries = {}

        encoded_columns: list[np.ndarray] = []

        for column_name in self.feature_names:
            series = self.raw_df[column_name]

            if pd.api.types.is_numeric_dtype(series):  # Si la columna es numérica no tiene sentido hacer el mapeo lexicográfico
                self.nominal_attributes.append(False)

                column_dict = {}
                self.dictionaries.append(column_dict)
                self.column_dictionaries[column_name] = column_dict

                # Guarda directamente el array numpy con los valores float
                encoded_columns.append(series.to_numpy(dtype=np.float64))
            else:  # Si es nominal realiza el mapeo lexicográfico
                self.nominal_attributes.append(True)

                # Obtiene los valores únicos (sin nulos) ordenados lexicográficamente
                unique_values = sorted(series.dropna().unique())
                column_dict = {val: idx for idx, val in enumerate(unique_values)}
                self.dictionaries.append(column_dict)
                self.column_dictionaries[column_name] = column_dict

                # Mapea los valores a su representación entera y convierte a float64
                encoded_columns.append(series.map(column_dict).to_numpy(dtype=np.float64))

        # Apila las columnas en un array de numpy 2D
        self.data = np.column_stack(encoded_columns)

    @property
    def num_samples(self) -> int:
        """
        Devuelve el número de muestras (filas) del dataset.

        Returns:
            int: Número de muestras.
        """
        return self.data.shape[0]

    @property
    def num_features(self) -> int:
        """
        Devuelve el número de atributos (columnas) del dataset.

        Returns:
            int: Número de atributos.
        """
        return self.data.shape[1] - 1  # Resta 1 por la última columna de etiquetas/clase objetivo

    @property
    def X(self) -> np.ndarray:
        """
        Devuelve la matriz de atributos (todas las columnas excepto la última).

        Returns:
            np.ndarray: Matriz de atributos.
        """
        return self.data[:, :-1]

    @property
    def y(self) -> np.ndarray:
        """
        Devuelve el vector de etiquetas de la clase (la última columna).

        Returns:
            np.ndarray: Vector de etiquetas.
        """
        return self.data[:, -1]

    def get_data(self, indices: list[int] | np.ndarray | None = None) -> np.ndarray:
        """
        Devuelve la matriz de datos numéricos procesada.

        Args:
            indices (list[int] | np.ndarray | None): Lista o array opcional de índices de filas a recuperar.

        Returns:
            np.ndarray: Matriz numpy 2D con las filas solicitadas.
        """
        return self.data[indices] if indices is not None else self.data

    def get_original_data(self, indices: list[int] | np.ndarray | None = None) -> pd.DataFrame:
        """
        Devuelve el DataFrame original sin codificar.

        Args:
            indices (list[int] | np.ndarray | None): Lista o array opcional de índices de filas a recuperar.

        Returns:
            pd.DataFrame: DataFrame de pandas con las filas solicitadas.
        """
        return self.raw_df.iloc[indices] if indices is not None else self.raw_df

    def __repr__(self) -> str:
        return (
            f"Dataset(file='{self.file_path}', samples={self.num_samples}, "
            f"features={self.data.shape[1]}, nominal_count={sum(self.nominal_attributes)})"
        )
