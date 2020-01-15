import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

n = int(input("n = "))
print(f'\nn = {n}')

mu = 67  # % 5'7''
sigma = 2.5  # 2 sigma = 5"

heights = mu * np.ones(n) + sigma * np.random.normal(0, 1, n)  # generate heights with gaussian distribution
# print(np.random.normal(0, 1, n))

marker_scaling = {50: 150, 100: 125, 500: 100, 1000: 75, 2500: 50, 9999999: 20}
marker_size = 50
for key, value in marker_scaling.items():  # defines how to scale the markers from the inputs
    if n <= key:
        marker_size = value
        break

# Plotting Scatter Plot
fig, ax = plt.subplots(figsize=(10, 10))
# ax = sns.scatterplot(np.arange(1, n + 1, 1), heights, marker="*", s=75, hue=heights, size=heights, sizes=(20, 200))  # generate x data starting from 1 to just before n+1, steps of 1
ax = sns.scatterplot(np.arange(1, n + 1, 1), heights, s=marker_size, hue=heights, legend='brief')
ax.set_title('Height vs. Student')
ax.set_xlabel('Student')
ax.set_ylabel('Height')
ax.axhline(y=mu, c='k', linewidth=3, label=r'$\mu = {{{}}}$'.format(mu))  # H-Line for mean

ax.axhline(y=mu + 2 * sigma, c='r', linestyle="dashed", label=r'$\mu \pm 2*\sigma$'.format(mu))  # H-Line for mean
ax.axhline(y=mu - 2 * sigma, c='r', linestyle="dashed", )  # H-Line for mean

ax.legend()  # places legend on right hand side
handles, labels = ax.get_legend_handles_labels()  # return labels and symbols
reverse_labels, reverse_handles = labels[2:], handles[2:]  # grab color symbols and labels
reverse_labels.reverse(), reverse_handles.reverse()  # reverse order, go from  high to low
labels[2:], handles[2:] = reverse_labels, reverse_handles  # reassign new order
# ax.legend(handles, labels, loc=2, ncol=1)
# ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True, shadow=True)
# fig.subplots_adjust(right=0.7)  # provides extra white space for legend on right hand side
plt.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('Data.png', bbox_inches='tight')
plt.show()

# Histogram and KDE at x_i of the Population
dx = (mu + 3 * sigma - mu + 3 * sigma) / 1000  # split entire distribution space evenly into 1000 parts
x = np.arange(mu - (3 * sigma), mu + (3 * sigma), dx)  # create a grid with even space of dx from +- 3sigma between mean
pdf = sp.stats.norm.pdf(x, loc=mu, scale=sigma)

# generate KDE from out sample data
kernel = sp.stats.gaussian_kde(heights)  # instantiate KDE method
test = kernel.evaluate(heights)
test2 = kernel(heights)

fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(x, pdf, label='Population PDF')
sns.kdeplot(heights, label='KDE', linestyle=':')
# plt.plot(np.arange(1, n + 1, 1), kernel(heights))
plt.title('Distribution Comparisons')
plt.xlabel('Height')
plt.axvline(x=mu, ymin=0, ymax=1, linestyle='--', alpha=0.25, label='Population Mean')
plt.ylim((0, 0.18))
plt.legend()
plt.savefig('PDF_and_KDES.png', bbox_inches='tight')
plt.show()

# plt.plot(np.arange(1, n + 1, 1), kernel(heights))
# plt.show()
