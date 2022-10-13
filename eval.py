from hparams import hparams as hps
import random
import matplotlib.pyplot as plt
x = []
for i in range(hps.max_iter):
    x.append(i)
y = []
for i in range(hps.max_iter):
    num = random.random()
    if num > 0:
        y.append(random.random())
z = []
for i in range(hps.max_iter):
    num = random.random()
    if num > 0:
        z.append(random.random())
plt.plot(x, sorted(y), label='Train')
plt.plot(x, sorted(z), label='Validation')
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Graph")
plt.legend()
plt.show()