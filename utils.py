import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import constantes


torch.manual_seed(constantes.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(constantes.seed)


def read_mnist_labels(filename):
    with open(filename, "rb") as f:
        # Leer el encabezado
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        # Asegurarse de que el archivo UBYTE tiene el formato esperado
        assert magic_number == 2049, "Magic number no coincide para etiquetas"

        # Leer las etiquetas
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def read_mnist_images(filename):
    with open(filename, "rb") as f:
        # Leer el encabezado
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        num_cols = int.from_bytes(f.read(4), byteorder="big")

        # Asegurarse de que el archivo UBYTE tiene el formato esperado
        assert magic_number == 2051, "Magic number no coincide"

        # Leer los datos de la imagen
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    return images


# Se agrega en el archivo para ejecutarlo una única vez y no pasarlo como hiperparámetro
x_train = torch.tensor(read_mnist_images('dataset/train-images-idx3-ubyte'), dtype=torch.float32) / 255.0
y_train = torch.tensor(read_mnist_labels('dataset/train-labels-idx1-ubyte'), dtype=torch.int64)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=564)
splits = list(sss.split(x_train, y_train))
train_idx, val_idx = splits[0]

x_temp = x_train
y_temp = y_train

x_train_aux, y_train_aux = x_temp[train_idx], y_temp[train_idx]
x_val, y_val = x_temp[val_idx], y_temp[val_idx]


def test_model(initial_model, loss_fn=nn.CrossEntropyLoss(), num_epochs=10, batch_sizes=[16],
               reducir_dataset=False, lrs=[0.01], verbose=True, l2_param=0.0, X_train=x_train_aux, Y_train=y_train_aux, X_val=x_val, Y_val=y_val):

    heatmap_data = []

    total_iterations = len(lrs) * len(batch_sizes)
    progress_bar = tqdm(total=total_iterations, desc="Cargando", position=0)

    models = []

    for lr in lrs:
        row_data = []
        for batch_size in batch_sizes:
            model = initial_model()
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_param)

            x_train_s, y_train_s = X_train, Y_train

            if reducir_dataset:
                x_train_s, y_train_s = select_instances(X_train, Y_train, num_instances_per_class=5)

            train_losses, train_accuracies, val_losses, val_accuracies = train_con_validacion(x_train_s, y_train_s, X_val, Y_val,
                                                                                  model, loss_fn,
                                                                                  optimizer, num_epochs=num_epochs,
                                                                                  batch_size=batch_size, verbose=verbose)

            if verbose:
                print('Resultado de usar lr', lr, 'y batch_size', batch_size)
                graficar(train_losses, val_losses, val_accuracies, train_accuracies)

            row_data.append(val_accuracies[-1])
            progress_bar.update(1)

            models.append((model, max(val_accuracies)))

        heatmap_data.append(row_data)

    if len(batch_sizes) > 1 or len(lrs) > 1:
        heatmap_data_transposed = np.transpose(heatmap_data)
        graficar_heatmap(heatmap_data_transposed, lrs, batch_sizes, num_epochs)

    progress_bar.close()

    return max(models, key=lambda x: x[1])


def train_con_validacion(x_train, y_train, x_val, y_val, model, loss_fn, optimizer, num_epochs=10, batch_size=64, verbose=True):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Crear dataloaders de entrenamiento
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Crear dataloaders de validación
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Almacenar la pérdida de entrenamiento
        train_losses.append(loss.item())

        # Calcular la precisión de entrenamiento
        model.eval()
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                total_train += batch_y.size(0)
                correct_train += predicted.eq(batch_y).sum().item()

        train_acc = 100. * correct_train / total_train
        train_accuracies.append(train_acc)

        # Validación
        correct_val = 0
        total_val = 0
        val_loss_items = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                val_outputs = model(batch_x)
                val_loss_items.append(loss_fn(val_outputs, batch_y).item())
                _, predicted = val_outputs.max(1)
                total_val += batch_y.size(0)
                correct_val += predicted.eq(batch_y).sum().item()

        val_loss = sum(val_loss_items) / len(val_loss_items)  # Calcular el promedio de las pérdidas de validación
        val_losses.append(val_loss)
        val_acc = 100. * correct_val / total_val
        val_accuracies.append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Training Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


def test(x_test, y_test, model):
    size = len(x_test)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        pred_probs = model(x_test)
        pred_labels = pred_probs.argmax(dim=1).cpu().numpy()  # Obtén las clases predichas y conviértelas a numpy
        y_true = y_test.cpu().numpy()  # Convierte y_test a numpy

        correct += (pred_labels == y_true).sum().item()

    accuracy = 100 * correct / size

    # Cálculo de métricas adicionales
    macro_f1 = f1_score(y_true, pred_labels, average='macro')
    precision = precision_score(y_true, pred_labels, average='macro')
    recall = recall_score(y_true, pred_labels, average='macro')

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro F1: {macro_f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    cm = confusion_matrix(y_true, pred_labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Matriz de confusión')
    plt.colorbar()
    num_classes = len(np.unique(y_true))
    plt.xticks(np.arange(num_classes), labels=constantes.PRENDAS, rotation=90)
    plt.yticks(np.arange(num_classes), labels=constantes.PRENDAS)
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color = 'red')

    plt.show()

    # Calcular el loss de cada imagen
    loss_values = nn.CrossEntropyLoss(reduction='none')(pred_probs, y_test).cpu().numpy()

    # Ordenar las imagenes por loss descendente
    sorted_indices = np.argsort(loss_values)[::-1]

    # Mostrar las 10 imagenes peor clasificadas
    plt.figure(figsize=(16, 10))
    plt.suptitle("Top 10 Imagenes Peor Clasificadas (Cross Entropy Loss)")

    for i in range(10):
        index = sorted_indices[i]
        plt.subplot(2, 5, i + 1)
        # Increase size
        plt.imshow(x_test[index], cmap='gray', )
        plt.title(
            f"Perdida: {loss_values[index]:.2f}\nPredijo: {constantes.PRENDAS[pred_labels[index]]} — Real: {constantes.PRENDAS[y_true[index]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def graficar_heatmap(data, lrs, batch_sizes, num_epochs):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='viridis', xticklabels=lrs, yticklabels=batch_sizes)
    plt.xlabel('lrs')
    plt.ylabel('batch_size')
    plt.title(f"Validation accuracies del último epoch para cada combinación de learning rates y batch sizes con {num_epochs} epochs")
    plt.gca().invert_yaxis()
    plt.show()


def graficar(train_losses, val_losses, val_accuracies, train_accuracies=None):
    # Graficar
    plt.figure(figsize=(12, 5))

    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss en entrenamiento', color='blue')
    plt.plot(val_losses, label='Loss en validación', color='green')
    plt.title('Loss sobre épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Accuracy en Validación', color='orange')

    if train_accuracies is not None:
        plt.plot(train_accuracies, label='Accuracy en Entrenamiento', color='red')

    plt.title('Accuracy sobre épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def select_instances(x, y, num_instances_per_class=100):
    selected_indices = []

    for label in torch.unique(y):
        label_indices = torch.where(y == label)[0]
        selected_label_indices = label_indices[:num_instances_per_class]
        selected_indices.extend(selected_label_indices.tolist())

    return x[selected_indices], y[selected_indices]


# Función para mostrar una imagen del tensor
def imshow(img_tensor, label):
    # Convertir el tensor a numpy
    img = img_tensor.numpy()

    # Mostrar la imagen con matplotlib
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.show()
