### Plot to make log pushback results graph ###
import matplotlib.pyplot as plt

labels = ["Toy", "Eco-small", "Eco-large"]
values = [(5/8) * 100, (5/5) * 100, (3/4) * 100]
plt.bar(labels, values, color=(0.1, 0.2, 0.6, 0.9))
plt.title('Log Pushback Accuracy')
plt.xlabel('Test example type')
plt.ylabel('Accuracy (%)')
plt.show()


### Plot to make storage/compute overhead results graph ###
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ["Squeezenet", "Resnet18", "Vgg16"]
naive_grad = [185732*10//1024, 219432*10//1024, 252878*10//1024]
naive_grad_plus_weight = [378450*10//1024, 461522*10//1024, 513365*10//1024]
hlast_grad = [185732*6//1024, 219432*6//1024, 252878*6//1024]
hlast_grad_plus_weight = [378450*6//1024, 461522*6//1024, 513365*6//1024]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * (width), naive_grad, width, label='Naive Grad')
rects2 = ax.bar(x - 0.5 * width, hlast_grad, width, label='HLAST Grad')
rects3 = ax.bar(x + 0.5 * (width), naive_grad_plus_weight, width, label='Naive Grad & Weights')
rects4 = ax.bar(x + 1.5 * width, hlast_grad_plus_weight, width, label='HLAST Grad & Weights')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Storage Overhead (in KB)')
ax.set_title('Storage Overhead for 100 epochs ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize = 'medium')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height-100),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()
plt.savefig(os.path.join("~/over_head_fig.png"))




