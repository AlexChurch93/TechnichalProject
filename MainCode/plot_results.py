import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab


# set the font
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 10}

plt.rc('font', **font)

def numpy_ewma_vectorized(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)    
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


acc_filename = 'Results/full_size_4000_acc.csv'
loss_filename = 'Results/full_size_4000_loss.csv'
lr_filename = 'Results/full_size_4000_lr.csv'

#loss_filename = 'Results/temp_loss.csv'

## ============== ACCURACY ==============
# import the csv file (change this to however you've saved it)
acc_file = pd.read_csv(acc_filename)

# Get the collumn names
acc_cols = acc_file.columns

# Change to simpler format
steps    = acc_file[acc_cols[1]]
accuracy = acc_file[acc_cols[2]]

#exponential smoothing
smooth_acc = numpy_ewma_vectorized(accuracy,10)

# third degree trendline
trendline = np.polyfit(steps,accuracy,2)
trendline_eq = np.poly1d(trendline)

acc_fig=pylab.figure()

# plot the lines
pylab.plot(steps,trendline_eq(steps),"r--")
pylab.plot(steps, accuracy, 'b-', alpha=0.2)
pylab.plot(steps, smooth_acc, 'b-')

# limit the axis
xlim = steps[len(steps)-1]+steps[1]
axes = plt.gca()
axes.set_xlim([0,xlim])
axes.set_ylim([0,1])

# label the plot
pylab.xlabel('Step')
pylab.ylabel('Batch Accuracy ')
pylab.title('Batch Training Accuracy : Final Acc = 0.89')

acc_fig.savefig(r"C:\Users\alexc\Documents\University Course Documents\Year 4\Technical Project\Images\acc.png", bbox_inches='tight', pad_inches=0)
pylab.show()


## ============== LOSS ==============
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)

# import the csv file (change this to however you've saved it)
loss_file = pd.read_csv(loss_filename)

# Get the collumn names
loss_cols = loss_file.columns

# Change to simpler format
steps = loss_file[loss_cols[1]]
loss  = loss_file[loss_cols[2]]

#exponential smoothing
smooth_loss = numpy_ewma_vectorized(loss,10)

loss_fig=pylab.figure()

# plot the lines
pylab.plot(steps, loss, '-', color='orange', linewidth=2)
#plt.plot(steps, smooth_loss, '-', color='orange')

# limit the axis
xlim = steps[len(steps)-1]+steps[1]
axes = plt.gca()
axes.set_xlim([0,xlim])
axes.set_ylim([0,50])

# label the plot
pylab.xlabel('Step')
pylab.ylabel('Batch Loss ')
pylab.title('Batch Mean Squared Error')

loss_fig.savefig(r"C:\Users\alexc\Documents\University Course Documents\Year 4\Technical Project\Images\loss.png",bbox_inches='tight', pad_inches=0)
pylab.show()


## ============== LEARNING RATE ==============
# import the csv file (change this to however you've saved it)
lr_file = pd.read_csv(lr_filename)

# Get the collumn names
lr_cols = lr_file.columns

# Change to simpler format
steps = lr_file[lr_cols[1]]
lr    = lr_file[lr_cols[2]]

lr_fig=pylab.figure()

# plot the lines
plt.plot(steps, lr, '-', color='green', linewidth=2)

# limit the axis
xlim = steps[len(steps)-1]+steps[1]
axes = plt.gca()
axes.set_xlim([0,xlim])
axes.set_ylim([0,0.001])

# label the plot
pylab.xlabel('Step')
pylab.ylabel('Learning Rate ')
pylab.title('Decaying Learning Rate')

lr_fig.savefig(r"C:\Users\alexc\Documents\University Course Documents\Year 4\Technical Project\Images\lr.png",bbox_inches='tight', pad_inches=0)
pylab.show()











