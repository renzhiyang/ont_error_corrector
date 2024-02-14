import os
import params
import argparse

import numpy as np
import matplotlib.pyplot as plt

from model import *
from datetime import datetime
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


parser = argparse.ArgumentParser(description="Filter candidate mutations")
parser.add_argument('--tensor_f', metavar='DIR', type=str, 
                    default="/home/yang1031/projects/ont_error_corrector/data/tensor_combined/hg002_1.tensor",
                    help='intermediate tensor file for NN models')
parser.add_argument('--model_path', metavar='DIR', type=str, 
                    default="/home/yang1031/projects/somatic/src/models/",
                    help='intermediate tensor file for NN models')
parser.add_argument('--out_predix', metavar='DIR', type=str, 
                    default="baseline",
                    help='intermediate tensor file for NN models')
parser.add_argument('--model', metavar='model_name', type=str, 
                    default="baseline",
                    help='resnet18, baseline')
parser.add_argument('--fp_weight', metavar='weight', type=float, 
                    default=1.0,
                    help='the weight that used to calculate weight of tumor categories in loss function')
parser.add_argument('--n_benchmark_variants', metavar='NUM', type=int, 
                    default=210000,
                    help='the number of benchmark variants in tensor_f')
parser.add_argument('--n_fp_variants', metavar='NUM', type=int,
                    default=33042,
                    help=' the number of false postive variants in tensor_f')
parser.add_argument('--model_name', metavar='MODEL', type=str,
                    default='DistilBert',
                    help=' the name of model: CNN_4layer, resnet8, DistilBert')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)

class VariantCallingDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.line_offsets = []
        self.total_lines = 0

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")

        with open(file_path, 'r') as file:
            offset = 0
            for line in file:
                if self.total_lines % self.chunk_size == 0:
                    self.line_offsets.append(offset)
                offset += len(line)
                self.total_lines += 1

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        line_idx = idx % self.chunk_size

        with open(self.file_path, 'r') as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()

            line = file.readline()
            parts = line.strip().split('\t')
            # Process the line to get data and labels as before
            # ... (data and label processing code goes here)
            if len(parts) != 5:
                print(f"warning: skipping malformed line {line}")
            
            # in label, 0 is benchmark calls, 1 is error calls
            # So after processing, benchmark call [1, 0], error calls [0, 1]
            label = [1.0, 0.0] if parts[3] == '0' else [0.0, 1.0]
            label = np.array(label, dtype=np.float32)
            input_data = [float(x) for x in parts[4].strip('[]').split(', ')]
            input_data = np.array(input_data).reshape(params.TENSOR_HEIGHT, params.TENSOR_WIDTH, params.TENSOR_CHANNELS)
            input_data = input_data.transpose(2, 0, 1)
            
            #exclude mapping quality
            
            input_data = input_data[:1, :, :]
            if args.model_name == 'DistilBert':
                input_data = input_data.reshape(params.TENSOR_HEIGHT, params.TENSOR_WIDTH)
                
            #print(input_data.shape, label.shape)
            #normalize data
            #for i in range(input_data.shape[0]):
            #    data_min = input_data[i].min()
            ##    data_max = input_data[i].max()
            #    input_data[i] = (input_data[i] - data_min) / (data_max - data_min)
            return input_data, label
          
                    
def loss_weights():    
    fp_ratio = args.fp_weight
    n_benchmark = args.n_benchmark_variants
    n_fp = args.n_fp_variants
    total_samples = n_benchmark + n_fp
    
    benchmark_weight = total_samples / n_benchmark
    fp_weight = fp_ratio * total_samples/ n_fp
    
    #convert to tensor
    return torch.FloatTensor([benchmark_weight, fp_weight])


def create_data_loaders(dataset, batch_size, train_ratio):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def train(model, dataloader, test_dataloader, criterion, optimizer, epochs):
    save_interval = 10
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= 300:
                break
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            
            #print(i, len(dataloader))
            torch.set_printoptions(threshold=10_000)
            #print(inputs[0, 3, 1:10, :])
            if args.model_name == 'resnet8':
                outputs = model(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #print(f'{i*params.BATCH_SIZE}/{len(dataloader)}')
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = \
            evaluate(model, test_dataloader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        print(f'Time:{datetime.now()}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%', flush=True)

        if (epoch + 1) % save_interval == 0:
            ensure_dir(args.model_path)
            model_save_path = os.path.join(args.model_path, f'{args.out_predix}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_save_path)
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f'Model saved at epoch {epoch+1}', flush=True)

            # Plotting
            plt.figure(figsize=(16, 12))
            plt.subplot(2, 3, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Over Time')
            plt.legend()

            plt.subplot(2, 3, 2)
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy Over Time')
            plt.legend()
            
            plt.subplot(2, 3, 3)
            plt.plot(val_precisions, label='Validation Precision')
            plt.xlabel('Epochs')
            plt.ylabel('Precision')
            plt.title('Validation Precision Over Time')
            plt.legend()

            plt.subplot(2, 3, 4)
            plt.plot(val_recalls, label='Validation Recall')
            plt.xlabel('Epochs')
            plt.ylabel('Recall')
            plt.title('Validation Recall Over Time')
            plt.legend()

            plt.subplot(2, 3, 5)
            plt.plot(val_f1s, label='Validation F1 Score')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.title('Validation F1 Score Over Time')
            plt.legend()

            plt.show()
            plot_save_path = os.path.join(args.model_path, f'{args.out_predix}_training_{epoch+1}epoch_plot.png')
            plt.savefig(plot_save_path)
            print(f'Plot saved at {plot_save_path}', flush=True)
        
    print(f'precisions: {val_precisions}', flush=True)
    print(f'recalls: {val_recalls}', flush=True)
    print(f'f1-scores: {val_f1s}', flush=True)
    # Plotting
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Time')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Validation Precision Over Time')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Validation Recall Over Time')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(val_f1s, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Over Time')
    plt.legend()


    plt.show()
    plot_save_path = os.path.join(args.model_path, f'{args.out_predix}_training_plot.png')
    plt.savefig(plot_save_path)
    print(f'Plot saved at {plot_save_path}', flush=True)
    return model           
 
def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total = 0
    correct = 0
    
    all_predicted = []
    all_labels = []

    with torch.no_grad():  # No gradient needed for evaluation
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= 100:
                break
            inputs, labels = inputs.float().to(device), labels.to(device)
            
            labels = labels.argmax(dim=1) # labels are one-hot encoded
            
            if args.model_name == 'resnet8':
                outputs = model(inputs)
            else:
                outputs = model(inputs)
            #print(f'outputs: {outputs}')
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(len(dataloader))
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate precision, recall and F1-score
    precision = precision_score(all_labels, all_predicted, average='weighted')
    recall = recall_score(all_labels, all_predicted, average='weighted')
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    return avg_loss, accuracy, precision, recall, f1

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)
    
    if args.model_name == 'DistilBert':
        model = CustomDistilBert(vocab_size=5, 
                                 num_features=9, 
                                 max_length=101, 
                                 num_classes=2,
                                 num_sequences=30).to(device)
        summary(model, input_size=(2, 30, 101))
    elif args.model_name == 'resnet6':
        layers = [2, 2, 2]
        model = ResNet_constant_kernel(BasicBlock_constant_kernel, layers).to(device)
        summary(model, input_size=(32, 1, 101, 30))
    elif args.model_name == 'CNN_6layer':
        model = CNN_6layer()
        summary(model, input_size=(32, 1, 101, 30))
    crossentropy_weights = loss_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=crossentropy_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    dataset = VariantCallingDataset(file_path=args.tensor_f, chunk_size=1000)
    train_loader, test_loader = create_data_loaders(dataset, params.BATCH_SIZE, params.TRAIN_RATIO)
    
    print(f"--------------Start training--------------", flush=True)
    trained_model = train(model, train_loader, test_loader, criterion, optimizer, epochs=100)
    
