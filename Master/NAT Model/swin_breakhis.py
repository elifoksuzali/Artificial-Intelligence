import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==================== GPU KONTROLÜ ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPU bulunamadı, CPU kullanılacak (yavaş olacak)")


model_checkpoint = "microsoft/swin-tiny-patch4-window7-224" 
batch_size = 32


from datasets import load_dataset

dataset = load_dataset("D:/archive (1)/BreaKHis_v1/BreaKHis_v1")

import evaluate

metric = evaluate.load("accuracy")
print("verımınız:",metric)

def simplify_label(example):
    path = example["image"].filename
    if "benign" in path.lower():
        example["label"] = 0
    else:
        example["label"] = 1
    return example

dataset = dataset.map(simplify_label)

id2label = {0: "benign", 1: "malignant"}
label2id = {"benign": 0, "malignant": 1}

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

from transformers import AutoImageProcessor

image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


splits = dataset["train"].train_test_split(test_size=0.3)
train_ds = splits['train']
temp_ds = splits['test']

val_test_splits = temp_ds.train_test_split(test_size=0.5)
val_ds = val_test_splits['train']
test_ds = val_test_splits['test']

print(f"📊 Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
test_ds.set_transform(preprocess_val)

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, 
)

# ==================== MODEL PARAMETERS ====================
print("\n" + "="*60)
print(" MODEL PARAMETERS")
print("="*60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f" Total Parameters: {total_params:,}")
print(f" Trainable Parameters: {trainable_params:,}")
print(f"  Non-Trainable (Frozen) Parameters: {non_trainable_params:,}")
print(f" Model Size: {total_params * 4 / (1024**2):.2f} MB (float32)")

print("\n Parameters by Module:")
print("-"*60)
for name, module in model.named_children():
    module_params = sum(p.numel() for p in module.parameters())
    module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name}: {module_params:,} params ({module_trainable:,} trainable)")

print("="*60)

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-breakhis",
    #report_to = "wandb",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # ===== GPU OPTİMİZASYONU =====
    fp16=torch.cuda.is_available(),  # Mixed precision (GPU varsa aktif)
    dataloader_pin_memory=True,       # Daha hızlı veri aktarımı
    dataloader_num_workers=2,         # Paralel veri yükleme
)

# ==================== HYPERPARAMETERS ====================
print("\n" + "="*60)
print("  HYPERPARAMETERS")
print("="*60)
print(f"  Model: {model_checkpoint}")
print(f" Batch Size (per device): {args.per_device_train_batch_size}")
print(f" Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
print(f" Number of Epochs: {args.num_train_epochs}")
print(f" Learning Rate: {args.learning_rate}")
print(f" Warmup Ratio: {args.warmup_ratio}")
print(f" Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
print(f" Logging Steps: {args.logging_steps}")
print(f" Metric for Best Model: {args.metric_for_best_model}")
print(f" Save Strategy: {args.save_strategy}")
print(f" Eval Strategy: {args.eval_strategy}")
print(f" FP16 (Mixed Precision): {args.fp16}")
print(f" Dataloader Pin Memory: {args.dataloader_pin_memory}")
print(f" Dataloader Num Workers: {args.dataloader_num_workers}")
print(f"  Image Size: {crop_size}")
print(f" Image Normalization Mean: {image_processor.image_mean}")
print(f" Image Normalization Std: {image_processor.image_std}")
print("="*60 + "\n")
# =========================================================

import numpy as np

import numpy as np

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == '__main__':
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,  
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Evaluate on test set
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)


    print("\n Generating training/validation loss graphs...")


    log_history = trainer.state.log_history

    train_loss = []
    train_steps = []
    eval_loss = []
    eval_epochs = []

    for log in log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_loss.append(log['loss'])
            train_steps.append(log.get('step', len(train_loss)))
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
            eval_epochs.append(log.get('epoch', len(eval_loss)))

    # Plot Training and Validation Loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Training Loss over Steps
    axes[0].plot(train_steps, train_loss, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Steps', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training vs Validation Loss over Epochs
    # Calculate average training loss per epoch for comparison
    steps_per_epoch = len(train_steps) // len(eval_epochs) if eval_epochs else 1
    train_loss_per_epoch = []
    for i in range(len(eval_epochs)):
        start_idx = i * steps_per_epoch
        end_idx = min((i + 1) * steps_per_epoch, len(train_loss))
        if start_idx < len(train_loss):
            epoch_avg = np.mean(train_loss[start_idx:end_idx])
            train_loss_per_epoch.append(epoch_avg)

    epochs = list(range(1, len(eval_loss) + 1))
    axes[1].plot(epochs, train_loss_per_epoch[:len(epochs)], 'b-o', linewidth=2, markersize=8, label='Training Loss')
    axes[1].plot(epochs, eval_loss, 'r-o', linewidth=2, markersize=8, label='Validation Loss')
    test_loss_value = test_metrics['eval_loss']
    axes[1].axhline(y=test_loss_value, color='green', linestyle='--', linewidth=2, label=f'Test Loss ({test_loss_value:.4f})')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training vs Validation vs Test Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)

    # Plot 3: Final Loss Comparison Bar Chart
    final_train_loss = train_loss_per_epoch[-1] if train_loss_per_epoch else train_loss[-1]
    final_val_loss = eval_loss[-1] if eval_loss else 0
    final_test_loss = test_loss_value
    
    loss_names = ['Train', 'Validation', 'Test']
    loss_values = [final_train_loss, final_val_loss, final_test_loss]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = axes[2].bar(loss_names, loss_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, loss_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Loss curves saved to 'loss_curves.png'")

    # ==================== CONFUSION MATRIX ====================
    print("\n Generating confusion matrix on test set...")


    predictions_output = trainer.predict(test_ds)
    predictions = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids


    cm = confusion_matrix(true_labels, predictions)
    class_names = ['Benign', 'Malignant']


    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 16}, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Confusion matrix saved to 'confusion_matrix.png'")

    # Print classification report
    print("\n Classification Report (Test Set):")
    print("=" * 50)
    print(classification_report(true_labels, predictions, target_names=class_names))

    # Print test loss
    print(f"\n Test Loss: {test_metrics['eval_loss']:.4f}")
    print(f" Test Accuracy: {test_metrics['eval_accuracy']:.4f}")