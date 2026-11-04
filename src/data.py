# src/data.py
import os
import numpy as np
import tensorflow as tf
from config import DATA_DIR, IMG_SIZE, EMOTIONS_EN, EMOTIONS_KZ, REPORTS_DIR

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(batch_size=64, val_split=0.15, seed=42):
    """train/val/test tf.data жиындарын қайтарады. Суреттер: (48, 48, 1), мәндер 0–255."""
    common = dict(
        labels="inferred",          # белгі = папка атауы
        label_mode="int",           # 0..6 бүтін сандар (sparse_categorical_crossentropy үшін)
        class_names=EMOTIONS_EN,    # класс ретін анық бекітеміз
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,            # батчты кейін, араластырғаннан соң жасаймыз
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), validation_split=val_split,
        subset="training", seed=seed, shuffle=True, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), validation_split=val_split,
        subset="validation", seed=seed, shuffle=True, **common)  # seed бірдей → жиындар қиылыспайды
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"), shuffle=False, **common)  # test ретін сақтаймыз (бағалау үшін)

    # cache: суреттерді жадта сақтау (FER шағын, ~100 МБ); shuffle: әр эпохада араластыру;
    # prefetch: GPU оқып жатқанда келесі батчты дайындау
    train_ds = train_ds.cache().shuffle(10_000, seed=seed).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def class_counts():
    return np.array([len(os.listdir(os.path.join(DATA_DIR, "train", e))) for e in EMOTIONS_EN])


def compute_class_weights():
    """Сирек класқа үлкен салмақ. Түбір (sqrt) салмақты тым асыра сілтемеуге көмектеседі:
    disgust үшін толық салмақ ≈ 9.4, ал sqrt-тен кейін ≈ 3.1."""
    counts = class_counts()
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    return {i: float(w) for i, w in enumerate(weights)}


if __name__ == "__main__":
    # 1-апта: деректерді зерттеу (EDA)
    import matplotlib.pyplot as plt
    os.makedirs(REPORTS_DIR, exist_ok=True)
    names = [EMOTIONS_KZ[e] for e in EMOTIONS_EN]
    counts = class_counts()
    for n, c in zip(names, counts):
        print(f"{n:10s} {c:5d}")

    plt.figure(figsize=(8, 4))
    plt.bar(names, counts, color="#3498db")
    plt.title("FER-2013 train: класс бойынша сурет саны")
    plt.ylabel("Сурет саны")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "class_distribution.png"), dpi=150)

    # Әр класстан 5 мысал
    train_ds, _, _ = load_datasets(batch_size=256)
    images, labels = next(iter(train_ds))
    fig, axes = plt.subplots(7, 5, figsize=(7, 10))
    for cls in range(7):
        idx = np.where(labels.numpy() == cls)[0][:5]
        for j, ax in enumerate(axes[cls]):
            ax.axis("off")
            if j < len(idx):
                ax.imshow(images[idx[j]].numpy().squeeze(), cmap="gray")
            if j == 0:
                ax.set_title(names[cls], fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "samples.png"), dpi=150)
    plt.show()
