# src/train.py
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from config import CNN_PATH, MOBILENET_PATH, MODELS_DIR, REPORTS_DIR
from data import load_datasets, compute_class_weights
from model_cnn import build_cnn
from model_transfer import build_mobilenet, unfreeze_top


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def get_callbacks(path, tag):
    return [
        keras.callbacks.BackupAndRestore(os.path.join(MODELS_DIR, "backup", tag)),
        keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]


def plot_history(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax in [("loss", ax1), ("accuracy", ax2)]:
        ax.plot(history[key], label="train")
        ax.plot(history["val_" + key], label="validation")
        ax.set_title(key); ax.set_xlabel("эпоха"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"history_{name}.png"), dpi=150)


def merge(*histories):
    out = {}
    for h in histories:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(REPORTS_DIR, exist_ok=True)
    keras.utils.set_random_seed(42)  # нәтижені қайталауға болатындай
    print("GPU:", tf.config.list_physical_devices("GPU"))

    train_ds, val_ds, test_ds = load_datasets(batch_size=args.batch)
    class_weight = compute_class_weights()
    print("Class weights:", class_weight)

    if args.model == "cnn":
        path = CNN_PATH
        model = build_cnn()
        compile_model(model, 1e-3)
        model.summary()
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                      class_weight=class_weight, callbacks=get_callbacks(path, "cnn"))
        history = merge(h)
    else:
        path = MOBILENET_PATH
        model, base = build_mobilenet()
        print("1-кезең: frozen base")
        compile_model(model, 1e-3)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=10,
                       class_weight=class_weight, callbacks=get_callbacks(path, "mobilenet_frozen"))
        print("2-кезең: fine-tuning")
        unfreeze_top(base, 30)
        compile_model(model, 1e-5)  # trainable өзгерген соң қайта compile міндетті!
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                       class_weight=class_weight, callbacks=get_callbacks(path, "mobilenet_finetune"))
        history = merge(h1, h2)

    plot_history(history, args.model)
    model.save(path)  # EarlyStopping ең жақсы салмақтарды қалпына келтірген
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"TEST accuracy: {acc:.3f}  (бұл санды тек соңында бір рет қараймыз)")


if __name__ == "__main__":
    main()
