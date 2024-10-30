import sys
sys.path.append('utils/')
from parsers import *

import os
from dotenv import load_dotenv     
load_dotenv('.env')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# for oncoplot side axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

data = pd.read_csv(os.environ.get("CANONICALSVFILE"), sep='\t')
data = data.copy().set_index('PUBLIC_ID')
data.columns = data.columns.str.split('_').str[2]

# Order the samples using the oncoplot algorithm
def oncoplot_ordering(data):
    # Define a recursive function to order the data
    def recursive_ordering(data, depth=0):
        if depth >= data.shape[1]:
            return data.index.tolist()
        
        # Order columns by column-wise sum
        primary_col = ordered_columns[depth]
        
        # Divide observations into groups based on the primary column
        group_1 = data[data[primary_col] == 1]
        group_0 = data[data[primary_col] == 0]
        
        # Recursively order within each group
        ordered_indices_1 = recursive_ordering(group_1, depth + 1)
        ordered_indices_0 = recursive_ordering(group_0, depth + 1)
        
        # Combine the ordered indices
        return ordered_indices_1 + ordered_indices_0
    
    # Start the recursive ordering
    ordered_columns = data.sum(axis=0).sort_values(ascending=False).index
    ordered_indices = recursive_ordering(data.loc[:, ordered_columns])
    return data.loc[ordered_indices, ordered_columns]


# Define a colormap with blue for 1 and white for 0
cmap = ListedColormap(['white', 'blue'])

ordered_data = oncoplot_ordering(data)
ordered_colnames = ordered_data.columns

plt.clf(); 
ax = plt.gca()
xlim = ax.set_xlim(0,ordered_data.shape[0])
ylim = ax.set_ylim(0,ordered_data.shape[1])

# instead of using axes.imshow, use axes.add_patch to draw rectangles
for i in range(ordered_data.shape[0]):
    for j in range(ordered_data.shape[1]):
        # draw a thin rectangle for each 1 in the data
        if ordered_data.iloc[i, j] == 1:
            ax.add_patch(
                plt.Rectangle(
                    (i, ordered_data.shape[1]-1-j), 1, 1, linewidth=0, color='blue'
                )
            )

# offset y-axis ticks to center them
ax.set_yticks(np.arange(len(ordered_colnames))+0.5)
# need to reverse the order of the labels this time
ax.set_yticklabels(ordered_colnames[::-1], rotation=0)

plt.title('RNA-Seq based IgH translocations')
plt.show()

plt.gcf().set_size_inches(8, 3)
plt.savefig('assets/heatmap_igh.png', dpi=150, bbox_inches='tight')
