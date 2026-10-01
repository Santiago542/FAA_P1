from abc import ABC, abstractmethod

import numpy as np

from preprocessing import Dataset


class Partition:
    """Representa una partición de datos, conteniendo los índices de entrenamiento y prueba."""

    def __init__(self, train_indices: list[int] | np.ndarray, test_indices: list[int] | np.ndarray):
        """
        Inicializa una partición con los índices de entrenamiento y prueba.

        Args:
            train_indices (list[int] | np.ndarray): Índices de entrenamiento.
            test_indices (list[int] | np.ndarray): Índices de prueba.
        """
        self.train_indices: np.ndarray = np.asarray(train_indices, dtype=np.int64)
        self.test_indices: np.ndarray = np.asarray(test_indices, dtype=np.int64)

    def __repr__(self) -> str:
        return f"Partition(train_size={len(self.train_indices)}, test_size={len(self.test_indices)})"


class PartitioningStrategy(ABC):
    """Estrategia abstracta para el particionado de datos."""

    def __init__(self):
        """Inicializa la estrategia de particionado."""
        self.partitions: list[Partition] = []

    @abstractmethod
    def create_partitions(self, dataset: Dataset, seed: int | None = None) -> list[Partition]:
        """
        Genera y almacena la lista de objetos Partition.

        Args:
            dataset (Dataset): Instancia de Dataset que proporciona el recuento de muestras.
            seed (int | None): Semilla aleatoria opcional para permutaciones reproducibles.

        Returns:
            list[Partition]: Lista de instancias de Partition.
        """


class SimpleValidation(PartitioningStrategy):
    """
    Divide los datos en un conjunto de entrenamiento y un conjunto de prueba basado en un porcentaje dado.
    Permite ejecutar múltiples permutaciones (ejecuciones) para calcular la media y la desviación estándar.
    """

    def __init__(self, test_percentage: float = 0.2, num_executions: int = 1):
        """
        Inicializa la estrategia SimpleValidation.

        Args:
            test_percentage (float): Fracción de muestras reservadas para prueba (entre 0.0 y 1.0).
            num_executions (int): Número de particiones aleatorias independientes a generar.
        """
        super().__init__()
        if not (0.0 < test_percentage < 1.0):
            raise ValueError("test_percentage debe estar estrictamente entre 0.0 y 1.0")
        if num_executions < 1:
            raise ValueError("num_executions debe ser >= 1")

        self.test_percentage: float = test_percentage
        self.num_executions: int = num_executions

    def create_partitions(self, dataset: Dataset, seed: int | None = None) -> list[Partition]:
        self.partitions = []
        n_samples = dataset.num_samples
        rng = np.random.default_rng(seed)  # Establece la semilla para reproducibilidad

        test_size = int(np.round(n_samples * self.test_percentage))
        train_size = n_samples - test_size

        for _ in range(self.num_executions):
            permuted = rng.permutation(n_samples)  # Baraja los índices para evitar sesgos
            train_idx = permuted[:train_size]
            test_idx = permuted[train_size:]

            self.partitions.append(Partition(train_idx, test_idx))

        return self.partitions


class CrossValidation(PartitioningStrategy):
    """
    Divide los datos en subconjuntos (folds) iguales (aproximadamente).
    Cada subconjunto se utiliza una vez como conjunto de prueba.
    Los restantes k-1 subconjuntos forman el conjunto de entrenamiento.
    """

    def __init__(self, k: int = 5):
        """
        Inicializa la estrategia CrossValidation.

        Args:
            k (int): Número de folds a crear (debe ser >= 2).
        """
        super().__init__()
        if k < 2:
            raise ValueError("k debe ser >= 2 para validación cruzada")
        self.k = k

    def create_partitions(self, dataset: Dataset, seed: int | None = None) -> list[Partition]:
        self.partitions = []
        n_samples = dataset.num_samples
        rng = np.random.default_rng(seed)  # Establece la semilla para reproducibilidad

        permuted = rng.permutation(n_samples)  # Baraja los índices para evitar sesgos

        folds = np.array_split(permuted, self.k)  # Divide los índices en k partes
        for fold_idx in range(self.k):
            # El fold excluido se convierte en el conjunto de prueba y el resto en el de entrenamiento
            train_folds = [folds[j] for j in range(self.k) if j != fold_idx]
            train_idx = np.concatenate(train_folds)
            test_idx = folds[fold_idx]

            self.partitions.append(Partition(train_idx, test_idx))

        return self.partitions
