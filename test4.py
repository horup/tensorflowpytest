import matplotlib.pyplot as plt
# plot a line, implicitly creating a subplot(111)
plt.subplot(221)
plt.plot([1, 2, 3, 3, 2, 1])
# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
plt.subplot(222)
plt.plot([1, 2, 3, 3, 2, 1])
plt.subplot(212)
plt.plot([1, 2, 3, 3, 2, 1])

plt.show()