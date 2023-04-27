
import numpy as np
import matplotlib.pyplot as plt

def neural_plot():
    inputLayerSize = 12
    outputLayerSize = 10

    selectedInput = ['w' for _ in range(inputLayerSize)]
    selectedOutput = ['w' for _ in range(outputLayerSize)]

    sIn = [1,4,5,10]
    sOut = [0,6,8]

    for s in sIn :
        selectedInput[s] = 'b'
    for s in sOut :
        selectedOutput[s] = 'r'
    W1 = np.random.randn(inputLayerSize,outputLayerSize)/2.5

    _, ax = plt.subplots()
    inputLayerX = np.arange(inputLayerSize) + 1
    outputLayerX = np.arange(outputLayerSize) + (inputLayerSize - outputLayerSize)/2 + 1
    ax.scatter(inputLayerX, np.zeros(inputLayerSize), s=100, c=selectedInput, marker='o', linewidths=1.5, edgecolors='k')
    ax.scatter(outputLayerX, np.zeros(outputLayerSize)+1, s=100, c=selectedOutput, marker='o', linewidths=1.5, edgecolors='k')
    for i in range(inputLayerSize):
        for j in range(outputLayerSize):
            if selectedInput[i] == 'b' and selectedOutput[j] == 'r':
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'black', linewidth=abs(W1[i,j]))
            else:
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'black', linewidth=abs(W1[i,j]))
    ax.set_xlim([0, inputLayerSize+1])
    ax.set_ylim([-0.5, 1.5])
    ax.axis('off')
    plt.show()

    _, ax = plt.subplots()
    inputLayerX = np.arange(inputLayerSize) + 1
    outputLayerX = np.arange(outputLayerSize) + (inputLayerSize - outputLayerSize)/2 + 1
    ax.scatter(inputLayerX, np.zeros(inputLayerSize), s=100, c=selectedInput, marker='o', linewidths=1.5, edgecolors='k')
    ax.scatter(outputLayerX, np.zeros(outputLayerSize)+1, s=100, c=selectedOutput, marker='o', linewidths=1.5, edgecolors='k')
    for i in range(inputLayerSize):
        for j in range(outputLayerSize):
            if selectedInput[i] == 'b' and selectedOutput[j] == 'r':
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'r', linewidth=abs(W1[i,j])*4)
            elif selectedInput[i] != 'b' and selectedOutput[j] == 'r':
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'green', linewidth=abs(W1[i,j])/2)
    ax.set_xlim([0, inputLayerSize+1])
    ax.set_ylim([-0.5, 1.5])
    ax.axis('off')
    plt.show()

    _, ax = plt.subplots()
    inputLayerX = np.arange(inputLayerSize) + 1
    outputLayerX = np.arange(outputLayerSize) + (inputLayerSize - outputLayerSize)/2 + 1
    ax.scatter(inputLayerX, np.zeros(inputLayerSize), s=100, c=selectedInput, marker='o', linewidths=1.5, edgecolors='k')
    ax.scatter(outputLayerX, np.zeros(outputLayerSize)+1, s=100, c=selectedOutput, marker='o', linewidths=1.5, edgecolors='k')
    for i in range(inputLayerSize):
        for j in range(outputLayerSize):
            if selectedInput[i] == 'b' and selectedOutput[j] == 'r':
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'r', linewidth=abs(W1[i,j])*4)
            elif selectedInput[i] != 'b' and selectedOutput[j] == 'r':
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'green', linewidth=abs(W1[i,j])/2)
            else :
                ax.plot([inputLayerX[i], outputLayerX[j]], [0, 1], 'k', linewidth=abs(W1[i,j]))
    ax.set_xlim([0, inputLayerSize+1])
    ax.set_ylim([-0.5, 1.5])
    ax.axis('off')
    plt.show()

def relu_gaussians():
    np.random.seed(0)
    g1 = np.random.normal(0, 1.2, 10000)
    g2 = np.random.normal(3, 1, 10000)
    g1,g2 = g1-1.6, g2-1.6
    g1 = np.maximum(0,g1)
    g2 = np.maximum(0,g2)
    counts1, bins1 = np.histogram(g1, bins = 100)
    counts2, bins2 = np.histogram(g2, bins = 100)
    plt.stairs(counts1, bins1)
    plt.stairs(counts2, bins2)
    plt.xlim([-1, 5])
    plt.show()

def cross_activation_diagram():
    inputLayerSize = 20
    outputLayerSize = 10

    selectedInput = ['w' for _ in range(inputLayerSize)]
    selectedOutput = ['w' for _ in range(outputLayerSize)]

    sIn = [1,4,5,10,13,18]
    sOut = [3]

    for s in sIn :
        selectedInput[s] = 'b'
    for s in sOut :
        selectedOutput[s] = 'r'
    W1 = np.random.randn(inputLayerSize,outputLayerSize)/2.5

    np.random.seed(0)
    g1 = np.random.normal(0, 0.5, 10000)
    g2 = np.random.normal(1.5, 0.4, 10000)
    counts1, bins1 = np.histogram(g1, bins = 100)
    counts2, bins2 = np.histogram(g2, bins = 100)
    _, ax = plt.subplots()
    
    # Add network diagram
    inputLayerSize = 20
    outputLayerSize = 10
    inputLayerX = np.arange(inputLayerSize) + 1
    outputLayerX = np.arange(outputLayerSize) + (inputLayerSize - outputLayerSize)/2 + 1
    ax.scatter(inputLayerX, np.zeros(inputLayerSize), s=100, c=selectedInput, marker='o', linewidths=1.5, edgecolors='k')
    ax.scatter(outputLayerX, np.zeros(outputLayerSize)+0.5, s=100, c=selectedOutput, marker='o', linewidths=1.5, edgecolors='k')
    ax.axis('off')
    ax.set_xlim([0, inputLayerSize+1])
    ax.set_ylim([-0.5, 1.])
    ax.set_position([0,0,1,1])
    
    ax2 = plt.axes([0.45, 0.04, 0.25, 0.22])
    ax2.stairs(counts1, bins1, label='Clean')
    ax2.stairs(counts2, bins2, label='Trigger')
    ax2.set_xlim([-1.3, 2.7])
    ax2.set_ylim([0, 400])
    # Add text (the labels) just above each graph line
    ax2.text(bins1[7]+0.82, counts1[7]+200, 'Clean', ha='center', va='bottom', color='blue', fontsize=10)
    ax2.text(bins2[5]+0.89, counts2[5]+200, 'Trigger', ha='center', va='bottom', color='orange', fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Add a line between the small graph, and a selected neuron of the first
    ax.plot([12.6, inputLayerX[13]], [-0.14, 0], 'black', linewidth=1.)
    #######
    np.random.seed(1)
    g1 = np.random.normal(1.5, 0.5, 10000)
    g2 = np.random.normal(0, 0.25, 10000)
    counts1, bins1 = np.histogram(g1, bins = 100)
    counts2, bins2 = np.histogram(g2, bins = 100)
    ax3 = plt.axes([0.23, 0.74, 0.25, 0.22])
    ax3.stairs(counts1, bins1, label='Clean')
    ax3.stairs(counts2, bins2, label='Trigger')
    ax3.set_xlim([-1.3, 2.7])
    ax3.set_ylim([0, 400])
    # Add text (the labels) just above each graph line
    ax3.text(bins1[7]+0.82, counts1[7]+200, 'Clean', ha='center', va='bottom', color='blue', fontsize=10)
    ax3.text(bins2[5]+0.2, counts2[5]+200, 'Trigger', ha='center', va='bottom', color='orange', fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])
    # Add a line between the small graph, and a selected neuron of the first
    ax.plot([8, outputLayerX[3]], [0.62, 0.5], 'black', linewidth=1.)

    ax.plot([inputLayerX[13], outputLayerX[3]], [0, 0.5], 'red', linewidth=2.5, linestyle='--')

    for s in sIn :
        if s != 13 :
            ax.plot([inputLayerX[s], outputLayerX[3]], [0, 0.5], 'red', linewidth=0.8)

    ax.text(11.9, 0.25, '$w<0$', ha='center', va='bottom', color='red', fontsize=10)

    plt.show()

#neural_plot()
#relu_gaussians()
#cross_activation_diagram()