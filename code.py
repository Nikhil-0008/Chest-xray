import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)
from tqdm import tqdm

TRAIN_DIR = r""
VAL_DIR = r""
TEST_DIR = r""

IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

OUTPUT_DIR = "GradCAM_Results"
ROBUSTNESS_DIR = "Robustness_Results"
FINAL_MODEL_NAME = "densenet121.h5"
BEST_CKPT_NAME = "densenet121_best.h5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)



train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.1,
    shear_range=0.05,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode="nearest"
)

eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_generator = eval_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

class_indices = train_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print("Class indices:", class_indices)

train_labels = train_generator.classes
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}
print("Class weights:", class_weight_dict)

#build
def build_model(unfreeze_last_n=30, dropout_rate=0.5):
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate, name="mc_dropout")(x)  # named so we can target it for MC-Dropout later
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model


model = build_model()
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(BEST_CKPT_NAME, monitor="val_auc", mode="max", save_best_only=True)
]


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train_loss")
axes[0].plot(history.history["val_loss"], label="val_loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(history.history["auc"], label="train_auc")
axes[1].plot(history.history["val_auc"], label="val_auc")
axes[1].set_title("AUC")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
plt.close()

# evaluation
test_generator.reset()
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn + 1e-10)   # recall for positive (PNEUMONIA) class
specificity = tn / (tn + fp + 1e-10)
precision = tp / (tp + fp + 1e-10)
f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)
roc_auc = roc_auc_score(y_true, y_pred_probs)
pr_auc = average_precision_score(y_true, y_pred_probs)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n=== Test Set Metrics ===")
print(f"Accuracy:    {accuracy*100:.2f}%")
print(f"Sensitivity (Recall): {sensitivity*100:.2f}%")
print(f"Specificity:          {specificity*100:.2f}%")
print(f"Precision:            {precision*100:.2f}%")
print(f"F1-score:             {f1*100:.2f}%")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print("\n", classification_report(y_true, y_pred, target_names=list(class_indices.keys())))

metrics_summary = {
    "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
    "precision": precision, "f1_score": f1, "roc_auc": roc_auc, "pr_auc": pr_auc
}
with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

# Confusion matrix plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(class_indices.keys()),
            yticklabels=list(class_indices.keys()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ROC and PR curves
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_probs)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

axes[1].plot(rec_curve, prec_curve, label=f"PR-AUC = {pr_auc:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_pr_curves.png"))
plt.close()


#gradcam
GRADCAM_LAYER_NAME = "relu"


def get_gradcam(model, img_array, last_conv_layer_name=GRADCAM_LAYER_NAME):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Binary sigmoid output -> single channel, use it directly
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10
    return heatmap.numpy()


def save_gradcam_overlays(model, generator, n_samples=40):
    print("Generating Grad-CAM overlays on a sample of test images...")
    generator.reset()
    results = []
    count = 0
    for batch_idx in tqdm(range(len(generator))):
        x_batch, y_batch = next(generator)
        preds = model.predict(x_batch, verbose=0).ravel()
        for i in range(len(x_batch)):
            if count >= n_samples:
                break
            img = np.expand_dims(x_batch[i], axis=0)
            heatmap = get_gradcam(model, img)
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


            original = x_batch[i]
            original = (original - original.min()) / (original.max() - original.min() + 1e-10)
            original = np.uint8(original * 255)

            overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

            pred_label = idx_to_class[int(preds[i] >= 0.5)]
            true_label = idx_to_class[int(y_batch[i])]
            pred_safe = sanitize_filename(pred_label)

            filename = f"test_img{count}_{pred_safe}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            results.append({
                "Image_File": filename,
                "Predicted_Label": pred_label,
                "True_Label": true_label,
                "Correct": pred_label == true_label
            })
            count += 1
        if count >= n_samples:
            break

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "gradcam_results.csv"), index=False)
    print(f"Grad-CAM results saved in '{OUTPUT_DIR}'")


save_gradcam_overlays(model, test_generator, n_samples=40)


def mc_dropout_predict(model, x_batch, n_iterations=30):

    preds = np.stack([
        model(x_batch, training=True).numpy().ravel()
        for _ in range(n_iterations)
    ], axis=0)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred


def run_uncertainty_analysis(model, generator, n_batches=10, n_iterations=30, high_unc_threshold=0.15):
    print("\nRunning MC-Dropout uncertainty estimation on test batches...")
    generator.reset()
    all_means, all_stds, all_true = [], [], []
    for b in tqdm(range(min(n_batches, len(generator)))):
        x_batch, y_batch = next(generator)
        mean_pred, std_pred = mc_dropout_predict(model, x_batch, n_iterations)
        all_means.extend(mean_pred)
        all_stds.extend(std_pred)
        all_true.extend(y_batch)

    df = pd.DataFrame({
        "true_label": all_true,
        "mean_prediction": all_means,
        "uncertainty_std": all_stds
    })
    df["predicted_label"] = (df["mean_prediction"] >= 0.5).astype(int)
    df["correct"] = df["predicted_label"] == df["true_label"]
    df["flagged_uncertain"] = df["uncertainty_std"] >= high_unc_threshold

    df.to_csv(os.path.join(OUTPUT_DIR, "uncertainty_results.csv"), index=False)

    n_flagged = df["flagged_uncertain"].sum()
    acc_confident = df.loc[~df["flagged_uncertain"], "correct"].mean() if (~df["flagged_uncertain"]).any() else float("nan")
    acc_uncertain = df.loc[df["flagged_uncertain"], "correct"].mean() if df["flagged_uncertain"].any() else float("nan")

    print(f"Flagged as high-uncertainty: {n_flagged}/{len(df)} samples")
    print(f"Accuracy on confident predictions: {acc_confident*100:.2f}%")
    print(f"Accuracy on flagged (uncertain) predictions: {acc_uncertain*100:.2f}%")
    return df


uncertainty_df = run_uncertainty_analysis(model, test_generator)


def add_gaussian_noise(img_uint8, sigma=15):
    noise = np.random.normal(0, sigma, img_uint8.shape).astype(np.float32)
    noisy = np.clip(img_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_gaussian_blur(img_uint8, ksize=5):
    return cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)


def change_contrast(img_uint8, factor=0.5):
    mean = img_uint8.mean()
    adjusted = (img_uint8.astype(np.float32) - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def jpeg_compress(img_uint8, quality=25):
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


PERTURBATIONS = {
    "gaussian_noise": lambda img: add_gaussian_noise(img, sigma=15),
    "gaussian_blur": lambda img: add_gaussian_blur(img, ksize=5),
    "low_contrast": lambda img: change_contrast(img, factor=0.5),
    "jpeg_compression": lambda img: jpeg_compress(img, quality=25),
}


def load_raw_test_images(test_dir, img_size=IMG_SIZE, max_per_class=150):
    """Load raw (uint8, un-preprocessed) test images directly from disk for
    perturbation testing, so degradations are applied in natural pixel space."""
    images, labels = [], []
    class_names = sorted(os.listdir(test_dir))
    for cls_name in class_names:
        cls_dir = os.path.join(test_dir, cls_name)
        files = os.listdir(cls_dir)[:max_per_class]
        for fname in files:
            path = os.path.join(cls_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(class_indices[cls_name])
    return np.array(images), np.array(labels)


def evaluate_on_images(model, raw_images, labels, batch_size=32):
    preprocessed = preprocess_input(raw_images.astype(np.float32).copy())
    probs = model.predict(preprocessed, batch_size=batch_size, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    acc = (preds == labels).mean()
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return acc, auc


def run_robustness_suite(model, test_dir):
    print("\nRunning robustness testing (noise / blur / contrast / compression)...")
    raw_images, labels = load_raw_test_images(test_dir)

    clean_acc, clean_auc = evaluate_on_images(model, raw_images, labels)
    results = [{"condition": "clean", "accuracy": clean_acc, "roc_auc": clean_auc}]
    print(f"Clean — Accuracy: {clean_acc*100:.2f}% | ROC-AUC: {clean_auc:.4f}")

    for name, perturb_fn in PERTURBATIONS.items():
        perturbed = np.array([perturb_fn(img) for img in raw_images])
        acc, auc = evaluate_on_images(model, perturbed, labels)
        results.append({"condition": name, "accuracy": acc, "roc_auc": auc})
        print(f"{name} — Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(ROBUSTNESS_DIR, "robustness_results.csv"), index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="condition", y="accuracy")
    plt.title("Robustness: Accuracy under Image Degradations")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(ROBUSTNESS_DIR, "robustness_accuracy.png"))
    plt.close()

    return results_df


robustness_df = run_robustness_suite(model, TEST_DIR)


model.save(FINAL_MODEL_NAME)
print(f"\n✅ Final model saved as {FINAL_MODEL_NAME}")
print(f"✅ Grad-CAM outputs saved in '{OUTPUT_DIR}'")
print(f"✅ Robustness outputs saved in '{ROBUSTNESS_DIR}'")
