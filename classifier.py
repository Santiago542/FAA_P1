from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from partitioning import PartitioningStrategy
from preprocessing import Dataset


class Classifier(ABC):
    """Clase abstracta para los clasificadores."""

    @abstractmethod
    def fit(self, train_data: np.ndarray, nominal_attributes: list[bool], dictionaries: list[dict[Any, int]]) -> Classifier:
        """
        Entrena el clasificador usando los datos de entrenamiento.

        Args:
            train_data (np.ndarray): Matriz de datos de entrenamiento (instancias + clase como última columna).
            nominal_attributes (list[bool]): Lista de booleanos indicando atributos nominales.
            dictionaries (list[dict[Any, int]]): Lista de diccionarios que mapean valores categóricos a enteros.

        Returns:
            Classifier: self
        """

    @abstractmethod
    def predict(self, test_data: np.ndarray, nominal_attributes: list[bool], dictionaries: list[dict[Any, int]]) -> np.ndarray:
        """
        Predice las etiquetas de la clase para las instancias de prueba.

        Args:
            test_data (np.ndarray): Matriz de datos de prueba (instancias + clase como última columna).
            nominal_attributes (list[bool]): Lista de booleanos indicando atributos nominales.
            dictionaries (list[dict[Any, int]]): Lista de diccionarios que mapean valores categóricos a enteros.

        Returns:
            np.ndarray: Array 1D con las etiquetas de clase predichas.
        """

    def error(self, actual: list[Any] | np.ndarray, predicted: list[Any] | np.ndarray) -> float:
        """
        Calcula la tasa de error de clasificación (proporción de instancias mal clasificadas).

        Args:
            actual (list[Any] | np.ndarray): Valores reales de la clase.
            predicted (list[Any] | np.ndarray): Valores predichos de la clase.

        Returns:
            float: Tasa de error entre 0.0 y 1.0.
        """
        actual_array = np.asarray(actual)
        predicted_array = np.asarray(predicted)

        if actual_array.shape != predicted_array.shape:
            raise ValueError(f"Discrepancia en las dimensiones: actual {actual_array.shape} vs predicho {predicted_array.shape}")

        if len(actual_array) == 0:
            return 0.0

        # actual_array != predicted_array genera un array de booleanos donde True es una instancia mal clasificada
        # np.mean calcula la media de los booleanos, que es la proporción de instancias mal clasificadas
        return float(np.mean(actual_array != predicted_array))

    def validate(self, partitioning_strategy: PartitioningStrategy, dataset: Dataset, seed: int | None = None) -> tuple[float, float, list[float]]:
        """
        Ejecuta el ciclo de validación completo.

        Args:
            partitioning_strategy (PartitioningStrategy): Estrategia para particionar el dataset.
            dataset (Dataset): Dataset a evaluar.
            seed (int | None): Semilla para la generación reproducible de particiones.

        Returns:
            tuple[float, float, list[float]]: (error_promedio, desviacion_estandar, lista_errores_particion)
        """
        partition_errors: list[float] = []

        partitioning_strategy.create_partitions(dataset, seed=seed)
        for partition in partitioning_strategy.partitions:
            train_subset = dataset.get_data(partition.train_indices)
            test_subset = dataset.get_data(partition.test_indices)

            # Entrena el modelo con los datos de entrenamiento y predice las etiquetas de la clase para los datos de prueba
            self.fit(train_subset, dataset.nominal_attributes, dataset.dictionaries)
            predictions = self.predict(test_subset, dataset.nominal_attributes, dataset.dictionaries)

            # Obtiene las etiquetas reales de la clase (última columna)
            actual_labels = test_subset[:, -1]

            # Calcula el error de la partición
            fold_err = self.error(actual_labels, predictions)
            partition_errors.append(fold_err)

        # Calcula el error promedio y la desviación estándar de las particiones
        mean_err = float(np.mean(partition_errors))
        std_err = float(np.std(partition_errors))

        return mean_err, std_err, partition_errors


class MajorityClassifier(Classifier):
    """Clasificador básico que siempre predice la clase más frecuente."""

    def __init__(self):
        self.majority_class: float | None = None

    def fit(self, train_data: np.ndarray, nominal_attributes: list[bool], dictionaries: list[dict[Any, int]]) -> MajorityClassifier:
        classes = train_data[:, -1]
        unique_classes, counts = np.unique(classes, return_counts=True)  # Obtiene los valores únicos y sus frecuencias
        self.majority_class = unique_classes[np.argmax(counts)]  # np.argmax devuelve el índice de la frecuencia más alta
        return self

    def predict(self, test_data: np.ndarray, nominal_attributes: list[bool], dictionaries: list[dict[Any, int]]) -> np.ndarray:
        """Genera un array con tantas filas como tenga el conjunto de test y rellenándolo solo con la clase mayoritaria."""
        return np.full(shape=(test_data.shape[0],), fill_value=self.majority_class, dtype=np.float64)
