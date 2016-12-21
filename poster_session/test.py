#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import nb_tweet as nb
import log_reg_improved as lr
import perceptron as pr

sizes = range(200,2900,200)
nb_accs = nb.limit_accuracy(sizes)
pr_accs = pr.limit_accuracy(sizes)
lr_accs = lr.limit_accuracy(sizes)

width = 0.2       # the width of the bars

fig, ax = plt.subplots()
ind = np.arange(len(sizes))  # the x locations for the groups
rects1 = ax.bar(ind - width, nb_accs, width, color='red', align='center')
rects2 = ax.bar(ind, pr_accs, width, color='green', align='center')
rects3 = ax.bar(ind + width, lr_accs, width, color='blue', align='center')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Size of Training Sample Set')
ax.set_xticks(ind)
ax.set_xticklabels(sizes)

ax.legend((rects1[0], rects2[0], rects3[0]), ('Naive Bayes', 'Perceptron', 'Log Reg'))

plt.show()

