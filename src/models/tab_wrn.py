"""
Tabular Wide ResNet-28 (TabWRN-28) architecture.

Faithfully implements the neural network architecture specified in Pisztora & Li (AAAI 2024):
"the architecture of the neural networks is the 'Wide ResNet-28' model (Zagoruyko and Komodakis 2016)
 adapted to tabular data with the replacement of convolutional layers with fully connected layers."

Depth = 28, consisting of:
- Initial linear layer mapping tabular input to base width
- 3 residual groups with N = (28 - 4) / 6 = 4 residual blocks per group
- Width multiplier k (e.g. k=2) across groups: [base_nodes * k, 2 * base_nodes * k, 4 * base_nodes * k]
- Batch normalization, ReLU activation, and dropout within residual blocks
- Final batch norm, ReLU, and linear projection head (with Softmax for classification)
"""

from typing import Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


class TabWRNResidualBlock(layers.Layer):
    """Single residual block of TabWRN-28 with pre-activation architecture."""

    def __init__(self, units: int, drop_rate: float = 0.1, weight_decay: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay

        reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.dense1 = layers.Dense(units, kernel_regularizer=reg)

        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.dropout = layers.Dropout(drop_rate)
        self.dense2 = layers.Dense(units, kernel_regularizer=reg)

        self.shortcut_dense = None

    def build(self, input_shape):
        in_dim = input_shape[-1]
        if in_dim != self.units:
            reg = regularizers.l2(self.weight_decay) if self.weight_decay > 0 else None
            self.shortcut_dense = layers.Dense(self.units, use_bias=False, kernel_regularizer=reg)
        super().build(input_shape)

    def call(self, inputs, training=False):
        shortcut = inputs
        if self.shortcut_dense is not None:
            shortcut = self.shortcut_dense(shortcut)

        x = self.bn1(inputs, training=training)
        x = self.relu1(x)
        x = self.dense1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        if training and self.drop_rate > 0:
            x = self.dropout(x, training=training)
        x = self.dense2(x)

        return shortcut + x


class TabWRNGroup(layers.Layer):
    """Group of N residual blocks in TabWRN-28."""

    def __init__(self, units: int, num_blocks: int = 4, drop_rate: float = 0.1, weight_decay: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            TabWRNResidualBlock(units, drop_rate=drop_rate, weight_decay=weight_decay)
            for _ in range(num_blocks)
        ]

    def call(self, inputs, training=False):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x


def build_tabwrn_model(
    input_dim: int,
    output_dim: int,
    task_type: str = "classification",
    base_nodes: int = 32,
    k: int = 2,
    drop_rate: float = 0.1,
    weight_decay: float = 1e-4,
) -> keras.Model:
    """
    Build TabWRN-28 Keras model.
    
    Structure:
    - Input (input_dim,)
    - Initial Dense projection -> base_nodes
    - Group 1: 4 blocks of width (base_nodes * k)
    - Group 2: 4 blocks of width (2 * base_nodes * k)
    - Group 3: 4 blocks of width (4 * base_nodes * k)
    - Post BN + ReLU
    - Output Dense (output_dim,) with Softmax for classification or Linear for regression.
    """
    inputs = layers.Input(shape=(input_dim,), name="tabular_input")
    reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

    # Initial projection
    x = layers.Dense(base_nodes, kernel_regularizer=reg, name="stem_dense")(inputs)

    # 3 groups with 4 blocks each (total 24 residual layers + stem + head = 28 layers)
    for g_idx in range(3):
        group_units = int(base_nodes * (2 ** g_idx) * k)
        group = TabWRNGroup(
            units=group_units,
            num_blocks=4,
            drop_rate=drop_rate,
            weight_decay=weight_decay,
            name=f"tabwrn_group_{g_idx}",
        )
        x = group(x)

    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(name="final_relu")(x)

    if task_type == "classification":
        outputs = layers.Dense(output_dim, activation="softmax", name="classification_head")(x)
    else:
        outputs = layers.Dense(output_dim, activation="linear", name="regression_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="TabWRN_28")
    return model


class TabWRNModel:
    """
    Sklearn-compatible wrapper for TabWRN-28.
    """

    def __init__(
        self,
        task_type: str = "classification",
        base_nodes: int = 32,
        k: int = 2,
        drop_rate: float = 0.1,
        weight_decay: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 40,
        random_state: int = 0,
    ):
        self.task_type = task_type
        self.base_nodes = base_nodes
        self.k = k
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.model: Optional[keras.Model] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 0,
    ) -> "TabWRNModel":
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X.shape[1]
        if self.task_type == "classification":
            output_dim = 2
            y_train_proc = keras.utils.to_categorical(y, num_classes=2)
            loss_fn = "categorical_crossentropy"
            metrics = ["accuracy"]
            val_data = (
                (X_val, keras.utils.to_categorical(y_val, num_classes=2))
                if X_val is not None
                else None
            )
        else:
            output_dim = 1
            y_train_proc = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            loss_fn = "mse"
            metrics = ["mse"]
            val_data = (
                (X_val, np.asarray(y_val, dtype=np.float32).reshape(-1, 1))
                if X_val is not None
                else None
            )

        self.model = build_tabwrn_model(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=self.task_type,
            base_nodes=self.base_nodes,
            k=self.k,
            drop_rate=self.drop_rate,
            weight_decay=self.weight_decay,
        )

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_data is not None else "loss",
                patience=10,
                restore_best_weights=True,
            )
        ]

        self.model.fit(
            X,
            y_train_proc,
            validation_data=val_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TabWRNModel has not been fitted.")
        preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        if self.task_type == "classification":
            return np.argmax(preds, axis=1)
        else:
            return preds.squeeze(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TabWRNModel has not been fitted.")
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification.")
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)
