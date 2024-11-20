import matplotlib.pyplot as plt
import json



def plot_training_history(history):
    epochs = range(len(history["train_loss"]))
    
    plt.figure(figsize=(12, 6))
    
    # График ошибок
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy During Training")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    with open("training_history.json", "r") as f:
        train_history = json.load(f)


    plot_training_history(train_history)



