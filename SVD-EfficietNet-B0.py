import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
import os
import gc

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Invalid device or cannot modify virtual devices once initialized.")

# Load and preprocess CIFAR-10 dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Convert data to float and normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Resize images for EfficientNet (requires 224x224)
    x_train_resized = tf.image.resize(x_train, (224, 224))
    x_test_resized = tf.image.resize(x_test, (224, 224))
    
    return x_train_resized, y_train, x_test_resized, y_test

# Build EfficientNet model
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# SVD Compression function
def apply_svd_compression(model, compression_ratio=0.5):
    compressed_model = tf.keras.models.clone_model(model)
    compressed_model.set_weights(model.get_weights())
    
    compressed_layers = []
    total_params_original = 0
    total_params_compressed = 0
    
    # Apply SVD to convolutional and dense layers
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) > 0:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Get layer weights
                W = weights[0]
                b = weights[1] if len(weights) > 1 else None
                
                # Count original parameters
                original_params = np.prod(W.shape)
                total_params_original += original_params
                
                # Reshape Conv2D weight for SVD
                original_shape = W.shape
                W_reshaped = W.reshape(original_shape[0] * original_shape[1] * original_shape[2], original_shape[3])
                
                # Apply SVD
                U, S, Vt = np.linalg.svd(W_reshaped, full_matrices=False)
                
                # Determine rank to keep based on compression ratio
                k = max(1, int(min(W_reshaped.shape) * compression_ratio))
                
                # Truncate matrices
                U_k = U[:, :k]
                S_k = np.diag(S[:k])
                Vt_k = Vt[:k, :]
                
                # Compute compressed weights
                W_compressed = U_k @ S_k @ Vt_k
                
                # Reshape back to original shape
                W_compressed = W_compressed.reshape(original_shape)
                
                # Count compressed parameters
                compressed_params = U_k.size + S_k.size + Vt_k.size
                total_params_compressed += compressed_params
                
                # Update weights in compressed model
                if b is not None:
                    compressed_model.layers[i].set_weights([W_compressed, b])
                else:
                    compressed_model.layers[i].set_weights([W_compressed])
                
                compressed_layers.append((layer.name, original_params, compressed_params))
                
            elif isinstance(layer, tf.keras.layers.Dense):
                # Get layer weights
                W = weights[0]
                b = weights[1] if len(weights) > 1 else None
                
                # Count original parameters
                original_params = W.size
                total_params_original += original_params
                
                # Apply SVD
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                
                # Determine rank to keep based on compression ratio
                k = max(1, int(min(W.shape) * compression_ratio))
                
                # Truncate matrices
                U_k = U[:, :k]
                S_k = np.diag(S[:k])
                Vt_k = Vt[:k, :]
                
                # Compute compressed weights
                W_compressed = U_k @ S_k @ Vt_k
                
                # Count compressed parameters
                compressed_params = U_k.size + S_k.size + Vt_k.size
                total_params_compressed += compressed_params
                
                # Update weights in compressed model
                if b is not None:
                    compressed_model.layers[i].set_weights([W_compressed, b])
                else:
                    compressed_model.layers[i].set_weights([W_compressed])
                
                compressed_layers.append((layer.name, original_params, compressed_params))
    
    return compressed_model, compressed_layers, total_params_original, total_params_compressed

# Evaluate models
def evaluate_models(original_model, compressed_model, x_test, y_test):
    # Evaluate original model
    start_time = time.time()
    original_loss, original_accuracy = original_model.evaluate(x_test, y_test, verbose=0)
    original_inference_time = time.time() - start_time
    
    # Evaluate compressed model
    start_time = time.time()
    compressed_loss, compressed_accuracy = compressed_model.evaluate(x_test, y_test, verbose=0)
    compressed_inference_time = time.time() - start_time
    
    # Get predictions for ROC curves
    original_preds = original_model.predict(x_test, verbose=0)
    compressed_preds = compressed_model.predict(x_test, verbose=0)
    
    return {
        'original': {
            'accuracy': original_accuracy,
            'loss': original_loss,
            'inference_time': original_inference_time,
            'predictions': original_preds
        },
        'compressed': {
            'accuracy': compressed_accuracy,
            'loss': compressed_loss,
            'inference_time': compressed_inference_time,
            'predictions': compressed_preds
        }
    }

# Calculate ROC curves and AUC
def calculate_roc_auc(y_true, y_pred):
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return fpr, tpr, roc_auc

# Plot evaluation metrics
def plot_results(evaluations, original_params, compressed_params, compression_ratios, class_names):
    # Set up the figure layout
    fig = plt.figure(figsize=(18, 12))
    
    # ROC curve - Micro average
    plt.subplot(2, 2, 1)
    for name, eval_data in evaluations.items():
        fpr, tpr, roc_auc = calculate_roc_auc(y_test, eval_data['predictions'])
        plt.plot(fpr['micro'], tpr['micro'], label=f'{name} (AUC = {roc_auc["micro"]:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Average ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # Comparison of parameters
    plt.subplot(2, 2, 2)
    models = list(evaluations.keys())
    params = [original_params, compressed_params]
    
    plt.bar(models, params)
    plt.ylabel('Number of Parameters')
    plt.title('Model Size Comparison')
    plt.grid(True)
    
    # Add parameter reduction percentage text
    reduction = (1 - compressed_params / original_params) * 100
    plt.text(1, compressed_params + max(params) * 0.05, f"{reduction:.1f}% reduction", ha='center')
    
    # Inference time comparison
    plt.subplot(2, 2, 3)
    inference_times = [eval_data['inference_time'] for eval_data in evaluations.values()]
    plt.bar(models, inference_times)
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time Comparison')
    plt.grid(True)
    
    # Add speedup percentage text
    speedup = (evaluations['original']['inference_time'] / evaluations['compressed']['inference_time'] - 1) * 100
    plt.text(1, inference_times[1] + max(inference_times) * 0.05, f"{speedup:.1f}% speedup", ha='center')
    
    # Accuracy comparison
    plt.subplot(2, 2, 4)
    accuracies = [eval_data['accuracy'] for eval_data in evaluations.values()]
    plt.bar(models, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim([0, 1])
    plt.grid(True)
    
    # Add accuracy difference text
    acc_diff = (evaluations['compressed']['accuracy'] - evaluations['original']['accuracy']) * 100
    plt.text(1, accuracies[1] + 0.05, f"{acc_diff:.2f}% difference", ha='center')
    
    plt.tight_layout()
    plt.savefig('svd_compression_results.png')
    plt.show()
    
    # Print results
    print(f"Compression Ratio: {compression_ratios}")
    print(f"Original Parameters: {original_params:,}")
    print(f"Compressed Parameters: {compressed_params:,}")
    print(f"Parameter Reduction: {reduction:.2f}%")
    print(f"Original Accuracy: {evaluations['original']['accuracy']:.4f}")
    print(f"Compressed Accuracy: {evaluations['compressed']['accuracy']:.4f}")
    print(f"Accuracy Difference: {acc_diff:.4f}%")
    print(f"Original Inference Time: {evaluations['original']['inference_time']:.4f}s")
    print(f"Compressed Inference Time: {evaluations['compressed']['inference_time']:.4f}s")
    print(f"Inference Speedup: {speedup:.2f}%")

# Main function
def main():
    print("Loading CIFAR-10 dataset...")
    x_train, y_train, x_test, y_test = load_data()
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("Building EfficientNetB0 model...")
    original_model = build_model()
    
    # Fine-tune the model with a small portion of data for demonstration
    # In practice, you'd train with more epochs and full dataset
    print("Fine-tuning the model on CIFAR-10...")
    # Use smaller batch size to reduce memory usage
    batch_size = 32
    # Use a small subset of data for demonstration
    subset_size = 5000
    original_model.fit(
        x_train[:subset_size], y_train[:subset_size],
        batch_size=batch_size,
        epochs=1,
        validation_data=(x_test[:1000], y_test[:1000]),
        verbose=1
    )
    
    # Free up memory
    gc.collect()
    if physical_devices:
        tf.keras.backend.clear_session()
    
    # Compression ratios to try
    compression_ratio = 0.5  # Keep 50% of singular values
    
    print(f"Applying SVD compression with ratio {compression_ratio}...")
    compressed_model, compressed_layers, original_params, compressed_params = apply_svd_compression(
        original_model, compression_ratio)
    
    # Evaluate models
    print("Evaluating models...")
    evaluations = {
        'original': {
            'accuracy': 0,
            'loss': 0,
            'inference_time': 0,
            'predictions': None
        },
        'compressed': {
            'accuracy': 0,
            'loss': 0,
            'inference_time': 0,
            'predictions': None
        }
    }
    
    # Evaluate in batches to save memory
    batch_size = 50
    num_batches = len(x_test) // batch_size
    
    # Original model evaluation
    start_time = time.time()
    original_preds = np.zeros((len(x_test), 10))
    original_loss = 0
    original_correct = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_x = x_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]
        
        preds = original_model.predict(batch_x, verbose=0)
        original_preds[start_idx:end_idx] = preds
        
        loss = tf.keras.losses.categorical_crossentropy(batch_y, preds).numpy().mean()
        original_loss += loss
        
        correct = np.sum(np.argmax(preds, axis=1) == np.argmax(batch_y, axis=1))
        original_correct += correct
        
    original_loss /= num_batches
    original_accuracy = original_correct / (num_batches * batch_size)
    original_inference_time = time.time() - start_time
    
    # Compressed model evaluation
    start_time = time.time()
    compressed_preds = np.zeros((len(x_test), 10))
    compressed_loss = 0
    compressed_correct = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_x = x_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]
        
        preds = compressed_model.predict(batch_x, verbose=0)
        compressed_preds[start_idx:end_idx] = preds
        
        loss = tf.keras.losses.categorical_crossentropy(batch_y, preds).numpy().mean()
        compressed_loss += loss
        
        correct = np.sum(np.argmax(preds, axis=1) == np.argmax(batch_y, axis=1))
        compressed_correct += correct
        
    compressed_loss /= num_batches
    compressed_accuracy = compressed_correct / (num_batches * batch_size)
    compressed_inference_time = time.time() - start_time
    
    evaluations = {
        'original': {
            'accuracy': original_accuracy,
            'loss': original_loss,
            'inference_time': original_inference_time,
            'predictions': original_preds
        },
        'compressed': {
            'accuracy': compressed_accuracy,
            'loss': compressed_loss,
            'inference_time': compressed_inference_time,
            'predictions': compressed_preds
        }
    }
    
    # Plot results
    print("Plotting results...")
    plot_results(evaluations, original_params, compressed_params, compression_ratio, class_names)
    
    # Print detailed layer compression information
    print("\nLayer-wise compression details:")
    print("-" * 80)
    print(f"{'Layer Name':<30} {'Original Params':<15} {'Compressed Params':<20} {'Compression Ratio':<15}")
    print("-" * 80)
    for layer_name, orig_params, comp_params in compressed_layers:
        ratio = comp_params / orig_params if orig_params > 0 else 0
        print(f"{layer_name:<30} {orig_params:,d} {comp_params:,d} {ratio:.4f}")

if __name__ == "__main__":
    main()