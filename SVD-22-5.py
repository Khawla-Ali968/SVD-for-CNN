import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import copy
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Loading and Preprocessing
def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),  # ResNet-50 requires 224x224 input
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),  # ResNet-50 requires 224x224 input
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# Load data
trainloader, testloader, classes = load_cifar10()

# Load pre-trained ResNet-50 model and adapt it for CIFAR-10
def get_resnet50_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# Function to perform SVD on convolutional layers
def apply_svd_to_conv(model, rank_percent=0.5):
    compressed_model = copy.deepcopy(model)
    
    # Dictionary to store original layer parameters for comparison
    original_params = {}
    compressed_params = {}
    
    for name, module in compressed_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip 1x1 convolutions as they're not suitable for this SVD approach
            if module.kernel_size == (1, 1):
                continue
                
            # Get the weights
            weight = module.weight.data.clone()
            original_params[name] = weight.numel()
            
            # Reshape the convolutional filter for SVD
            out_channels, in_channels, kh, kw = weight.size()
            weight_reshaped = weight.reshape(out_channels, -1)
            
            # Perform SVD
            U, S, V = torch.svd(weight_reshaped)
            
            # Calculate how many singular values to keep
            k = max(1, int(min(weight_reshaped.size()) * rank_percent))
            
            # Truncate the matrices
            U_truncated = U[:, :k]
            S_truncated = torch.diag(S[:k])
            V_truncated = V[:, :k]
            
            # Reconstruct the weights
            weight_reconstructed = torch.mm(torch.mm(U_truncated, S_truncated), V_truncated.t())
            weight_reconstructed = weight_reconstructed.reshape(out_channels, in_channels, kh, kw)
            
            # Update the weights
            module.weight.data = weight_reconstructed
            
            # Count parameters in compressed representation
            compressed_params[name] = U_truncated.numel() + S_truncated.numel() + V_truncated.numel()
            
    return compressed_model, original_params, compressed_params

# Fine-tune the compressed model
def fine_tune(model, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")
    
    return model

# Evaluate model accuracy
def evaluate_model(model, testloader):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Measure inference time
def measure_inference_time(model, testloader, num_batches=100):
    model.to(device)
    model.eval()
    
    # Warm-up
    for i, data in enumerate(testloader):
        if i >= 5:
            break
        images, _ = data
        images = images.to(device)
        _ = model(images)
    
    # Measure time
    start_time = time.time()
    batch_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if batch_count >= num_batches:
                break
                
            images, _ = data
            images = images.to(device)
            _ = model(images)
            batch_count += 1
    
    end_time = time.time()
    inference_time = (end_time - start_time) / batch_count
    
    return inference_time

# Calculate model size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb

# Main execution
def main():
    # Load and prepare models
    original_model = get_resnet50_model()
    original_model.to(device)
    
    print("Original model loaded")
    original_size = get_model_size(original_model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Test with different SVD compression ratios
    compression_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for ratio in compression_ratios:
        print(f"\nApplying SVD with compression ratio: {ratio}")
        compressed_model, orig_params, comp_params = apply_svd_to_conv(get_resnet50_model(), ratio)
        
        # Calculate parameter reduction
        total_original_params = sum(orig_params.values())
        total_compressed_params = sum(comp_params.values())
        param_reduction = 1 - (total_compressed_params / total_original_params)
        
        # Fine-tune the compressed model
        compressed_model = fine_tune(compressed_model, trainloader, epochs=3)
        
        # Evaluate the compressed model
        compressed_accuracy = evaluate_model(compressed_model, testloader)
        
        # Measure inference time
        original_inference_time = measure_inference_time(original_model, testloader)
        compressed_inference_time = measure_inference_time(compressed_model, testloader)
        speedup = original_inference_time / compressed_inference_time
        
        # Calculate compression metrics
        compressed_size = get_model_size(compressed_model)
        compression_ratio_actual = original_size / compressed_size
        
        # Store results
        results.append({
            'ratio': ratio,
            'param_reduction': param_reduction * 100,  # as percentage
            'speedup': speedup,
            'compression_ratio': compression_ratio_actual,
            'accuracy': compressed_accuracy,
            'model_size_mb': compressed_size
        })
        
        print(f"Compression Results for ratio {ratio}:")
        print(f"Parameter Reduction: {param_reduction * 100:.2f}%")
        print(f"Inference Speedup: {speedup:.2f}x")
        print(f"Size Compression Ratio: {compression_ratio_actual:.2f}x")
        print(f"Accuracy: {compressed_accuracy:.2f}%")
        print(f"Model Size: {compressed_size:.2f} MB")
    
    # Get original model accuracy for comparison
    print("\nEvaluating original model...")
    original_accuracy = evaluate_model(original_model, testloader)
    
    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract results for plotting
    ratios = [r['ratio'] for r in results]
    param_reductions = [r['param_reduction'] for r in results]
    speedups = [r['speedup'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Parameter Reduction
    axs[0, 0].plot(ratios, param_reductions, 'o-', color='blue')
    axs[0, 0].set_xlabel('SVD Rank Percentage')
    axs[0, 0].set_ylabel('Parameter Reduction (%)')
    axs[0, 0].set_title('Parameter Reduction vs SVD Rank')
    axs[0, 0].grid(True)
    
    # Inference Speedup
    axs[0, 1].plot(ratios, speedups, 'o-', color='green')
    axs[0, 1].set_xlabel('SVD Rank Percentage')
    axs[0, 1].set_ylabel('Speedup Factor')
    axs[0, 1].set_title('Inference Speedup vs SVD Rank')
    axs[0, 1].grid(True)
    
    # Compression Ratio
    axs[1, 0].plot(ratios, compression_ratios, 'o-', color='red')
    axs[1, 0].set_xlabel('SVD Rank Percentage')
    axs[1, 0].set_ylabel('Compression Ratio')
    axs[1, 0].set_title('Compression Ratio vs SVD Rank')
    axs[1, 0].grid(True)
    
    # Accuracy
    axs[1, 1].plot(ratios, accuracies, 'o-', color='purple')
    axs[1, 1].axhline(y=original_accuracy, color='r', linestyle='--', label=f'Original ({original_accuracy:.2f}%)')
    axs[1, 1].set_xlabel('SVD Rank Percentage')
    axs[1, 1].set_ylabel('Accuracy (%)')
    axs[1, 1].set_title('Accuracy vs SVD Rank')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('svd_compression_results.png')
    plt.show()
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df['original_accuracy'] = original_accuracy
    results_df['original_size_mb'] = original_size
    results_df.to_csv('svd_compression_results.csv', index=False)
    
    print("\nResults saved to svd_compression_results.csv and svd_compression_results.png")

if __name__ == "__main__":
    main()