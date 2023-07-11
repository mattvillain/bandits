import matplotlib.pyplot as plt
import numpy as np
def plot_threeway_combinations(x1,x2,x3,x4, y):
# Compute the output based on the input mapping (Example function)
    # Plotting
    fig = plt.figure(figsize=(12, 8))

    # Scatter plot of the input variables (x1, x2, x3)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x1, x2, x3, c=y, cmap='viridis')
    ax1.set_xlabel('Sepal Length')
    ax1.set_ylabel('Sepal Width')
    ax1.set_zlabel('Petal Length')
    ax1.set_title('Scatter plot of Sepal Length, Width and Petal Length with color-coded y')

    # Scatter plot of the input variables (x1, x3, x4)
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(x1, x3, x4, c=y, cmap='viridis')
    ax2.set_xlabel('Sepal Length')
    ax2.set_ylabel('Petal Length')
    ax2.set_zlabel('Petal Width')
    ax2.set_title('Scatter plot of Sepal Length, Petal Length, Width with color-coded y')

    # Scatter plot of the input variables (x2, x3, x4)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(x2, x3, x4, c=y, cmap='viridis')
    ax3.set_xlabel('Sepal Width')
    ax3.set_ylabel('Petal Length')
    ax3.set_zlabel('Petal Width')
    ax3.set_title('Scatter plot of Sepal Width, Petal Length, Width with color-coded y')

    # Histogram of the output variable (y)
    ax4 = fig.add_subplot(224)
    ax4.hist(y, bins=20, edgecolor='black')
    ax4.set_xlabel('y')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Histogram of y')

    plt.tight_layout()
    plt.show()