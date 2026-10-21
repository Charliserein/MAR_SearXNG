import json
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def map_to_binary(label):
    """
    - 'original' -> 0 (real)
    - other categories -> 1 (fake)
    """
    if label == "original":
        return 0
    else:
        return 1


def evaluate_binary_classification(true_labels, pred_labels):

    true_binary = [map_to_binary(label) for label in true_labels]
    pred_binary = [map_to_binary(label) for label in pred_labels]

    return calculate_multiclass_metrics(true_binary, pred_binary, 2)


def evaluate_multiclass_classification(true_labels, pred_labels):

    all_classes = sorted(list(set(true_labels).union(set(pred_labels))))
    num_classes = len(all_classes)
    
    print(f"\nDetected {num_classes} category: {all_classes}")
    
    return calculate_multiclass_metrics(true_labels, pred_labels, num_classes, all_classes)


def calculate_multiclass_metrics(y_true, y_pred, actual_class, custom_labels=None):

    if custom_labels is not None:
        all_labels = custom_labels
    elif actual_class == 2:
        all_labels = [0, 1]
    else:
        all_labels = sorted(list(set(y_true).union(set(y_pred))))

    num_classes = len(all_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    for t, p in zip(y_true, y_pred):
        true_index = label_to_index[t]
        pred_index = label_to_index[p]
        conf_matrix[true_index][pred_index] += 1

    # accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # precision
    precision_per_class = precision_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    precision = np.sum(precision_per_class) / actual_class

    # recall
    recall_per_class = recall_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    recall = np.sum(recall_per_class) / actual_class

    # f1
    f1_per_class = f1_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    f1 = np.sum(f1_per_class) / actual_class

    return {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "all_labels": all_labels,
        "num_classes": num_classes
    }


def load_results_from_json(file_path):

    true_labels = []
    pred_labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                gt = data["fake_cls"]
                pred = data["label"]

                true_labels.append(gt)
                pred_labels.append(pred)

            except Exception as e:
                print(f"{line_num} line error: {e}")
                continue

    return true_labels, pred_labels


def print_binary_results(results):

    print("\n" + "="*60)
    print("Binary classification evaluation results")
    print("="*60)
    print(f"accuracy (Accuracy): {results['accuracy']:.4f}")
    print(f"f1_score (F1-Score): {results['f1_score']:.4f}")
    print(f"precision (Precision): {results['precision']:.4f}")
    print(f"recall (Recall): {results['recall']:.4f}")
    print("\confusion_matrix:")
    print("real\\prediction   0(real)    1(fake)")
    print(f"0(real)    {results['confusion_matrix'][0][0]:8d}    {results['confusion_matrix'][0][1]:8d}")
    print(f"1(fake)    {results['confusion_matrix'][1][0]:8d}    {results['confusion_matrix'][1][1]:8d}")


def print_multiclass_results(results):
    """
    Print the evaluation results of multi-class classification
    """
    num_classes = results['num_classes']
    all_labels = results['all_labels']
    
    print("\n" + "="*60)
    print(f"{num_classes} classification")
    print("="*60)
    print(f"accuracy (Accuracy): {results['accuracy']:.4f}")
    print(f"f1_score (F1-Score): {results['f1_score']:.4f}")
    print(f"precision (Precision): {results['precision']:.4f}")
    print(f"recall (Recall): {results['recall']:.4f}")

    print("\confusion_matrix:")
    
    max_label_len = max(len(str(label)) for label in all_labels)
    col_width = max(8, max_label_len + 1)
    
    print("real\\prediction".ljust(max_label_len), end="")
    for label in all_labels:
        label_str = str(label)[:col_width-1]
        print(f"{label_str:>{col_width}}", end="")
    print()

    for i, true_label in enumerate(all_labels):
        true_label_str = str(true_label)[:max_label_len]
        print(f"{true_label_str:>{max_label_len}}", end="")
        for j in range(num_classes):
            print(f"{results['confusion_matrix'][i][j]:>{col_width}}", end="")
        print()


def print_label_distribution(true_labels, pred_labels):

    print("\n" + "="*60)
    print("Tag Distribution Statistics")
    print("="*60)

    true_counts = {}
    pred_counts = {}

    for label in true_labels:
        true_counts[label] = true_counts.get(label, 0) + 1

    for label in pred_labels:
        pred_counts[label] = pred_counts.get(label, 0) + 1

    print("Actual label distribution:")
    for label, count in sorted(true_counts.items()):
        percentage = count / len(true_labels) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print("\nPredict label distribution:")
    for label, count in sorted(pred_counts.items()):
        percentage = count / len(pred_labels) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")


def main():

    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--model', type=str, default='Qwen3-VL-32B-Instruct-FP8', help='model name')
    parser.add_argument('--output_dir', type=str, default='../dataset_final/MMFakeBench', help='Output directory path')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'test'], default='train', help='dataset_type: train or test')

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model)
    output_file = os.path.join(output_dir, f"{args.dataset_type}_mmfakebench_32b.json")

    if not os.path.exists(output_file):
        print(f"Error: Unable to find the file  {output_file}")
        return

    print(f"The document is currently under evaluation: {output_file}")

    true_labels, pred_labels = load_results_from_json(output_file)

    if len(true_labels) == 0:
        print("Error")
        return

    print(f"Successfully loaded {len(true_labels)} samples")

    print_label_distribution(true_labels, pred_labels)

    # Binary
    print("\n" + "="*60)
    print("1: Binary")
    print("="*60)
    binary_results = evaluate_binary_classification(true_labels, pred_labels)
    print_binary_results(binary_results)

    # Multi-class evaluation
    print("\n" + "="*60)
    print("2: Multi-class evaluation")
    print("="*60)
    multiclass_results = evaluate_multiclass_classification(true_labels, pred_labels)
    print_multiclass_results(multiclass_results)

    results_summary = {
        'dataset_type': args.dataset_type,
        'model': args.model,
        'total_samples': len(true_labels),
        'binary_classification': {
            'accuracy': float(binary_results['accuracy']),
            'f1_score': float(binary_results['f1_score']),
            'precision': float(binary_results['precision']),
            'recall': float(binary_results['recall']),
            'confusion_matrix': binary_results['confusion_matrix'].tolist()
        },
        'multiclass_classification': {
            'num_classes': multiclass_results['num_classes'],
            'class_labels': multiclass_results['all_labels'],
            'accuracy': float(multiclass_results['accuracy']),
            'f1_score': float(multiclass_results['f1_score']),
            'precision': float(multiclass_results['precision']),
            'recall': float(multiclass_results['recall']),
            'confusion_matrix': multiclass_results['confusion_matrix'].tolist()
        }
    }

    eval_output_file = os.path.join(output_dir, f"{args.dataset_type}_result_mmfakebench_32b.json")
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nThe assessment results have been saved to: {eval_output_file}")


if __name__ == "__main__":
    main()