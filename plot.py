### Plot to make log pushback results graph ###
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ["Toy", "Random", "Eco-small", "Eco-large"]
values_gumtree = [round((4/7) * 100, 2), round((0) * 100, 2), round((4/5) * 100, 2), round((0) * 100, 2)]
values_dmp = [round((5/8) * 100, 2), round((0) * 100, 2), round((5/5) * 100, 2), round((3/4) * 100, 2)]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values_gumtree, width, label='GumTree AST Diff')
rects2 = ax.bar(x + width/2, values_dmp, width, label='Myers Algo with DMP')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Log Pushback Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize = 'medium')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height-2),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig(os.path.join("~/logpushback_fig.png"))


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
plt.savefig(os.path.join("~/overhead_fig.png"))




