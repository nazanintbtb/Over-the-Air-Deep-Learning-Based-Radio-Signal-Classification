import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import load_data as loader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curve(y_true, y_score, classes):
    plt.figure(figsize=(10, 10))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve for {classes[i]} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(y_true, y_score, classes):
    plt.figure(figsize=(10, 10))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'PR curve for {classes[i]} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

classes=["PAM4","QAM16","QAM64","BPSK","8BPS","QPS","CPFS","GFSK"]
model_path = 'test.h5'
model = load_model(model_path)
test_Y_hat = model.predict(loader.get_test_data(10))

data=loader.get_test_data(634)
for batch in data.take(1):
    x_test, y_true = batch

print(test_Y_hat.shape)
print(y_true.shape)
# print()

conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,x_test.shape[0]):
    j = list(y_true[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
#
plot_confusion_matrix(confnorm, labels=classes)
plot_roc_curve(y_true, test_Y_hat, classes)
plot_pr_curve(y_true, test_Y_hat, classes)
for i in range(len(confnorm)):
    print(classes[i],confnorm[i,i])