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
from matplotlib.patches import Patch

data = pd.read_csv(os.environ.get("CANONICALSVFILE"), sep='\t')
data = data.copy().set_index('PUBLIC_ID')
data.columns = data.columns.str.split('_').str[2]

# categorical data for top/bottom axes
extra_covariates = ['ISS','OLD','SEX']
side = pd.read_csv(os.environ.get("CLINDATAFILE"), sep='\t')\
    .set_index('PUBLIC_ID')\
    .merge(data, left_index=True, right_index=True, how='right')\
    .assign(ISS = lambda df: df['D_PT_iss'].fillna(4).astype('int'),
            SEX = lambda df: df['D_PT_gender'].fillna(3).astype('int') - 1,
            OLD = lambda df: (df['D_PT_age'] >= 63).fillna(2).astype('int'))\
    .loc[lambda df: df.index.isin(data.index), :]\
    [extra_covariates]
    
full_data = pd.concat([data,side],axis=1)
# full_data = data

# Order the samples using the oncoplot algorithm
def oncoplot_ordering(data):
    # Define a recursive function to order the data
    def recursive_ordering(data, depth=0):
        if depth >= data.shape[1]:
            return data.index.tolist()
        
        # Order columns by column-wise sum
        primary_col = ordered_columns[depth]
                
        # The remaining are IGH partner columns
        # Divide observations into groups based on the primary column
        
        ordered_indices = []
        
        for value in sorted(data[primary_col].unique())[::-1]:
            # subset to observations with the same value
            group_v = data[data[primary_col] == value]
            # Recursively order within each group
            ordered_indices_v = recursive_ordering(group_v, depth + 1)
            # Combine the ordered indices
            ordered_indices.extend(ordered_indices_v)
        
        return ordered_indices
    
    # Start the recursive ordering
    ordered_columns = data.sum(axis=0).sort_values(ascending=False).index
    
    # Order additional columns appear last
    ordered_columns = ordered_columns.drop(extra_covariates)
    ordered_columns = ordered_columns.append(pd.Index(extra_covariates))
    ordered_indices = recursive_ordering(data.loc[:, ordered_columns])
    return data.loc[ordered_indices, ordered_columns]


ordered_data_full = oncoplot_ordering(full_data)
# should be 875 x 11
 
ordered_colnames = ordered_data_full.columns[:-3]
clin_colnames = ordered_data_full.columns[-3:]
ordered_data = ordered_data_full[ordered_colnames]
ordered_clin = ordered_data_full[clin_colnames]

# ordered_data = ordered_data_full
# ordered_colnames = ordered_data_full.columns 
# ordered_clin = side 
# clin_colnames = side.columns


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
                    (i, ordered_data.shape[1]-1-j), 1, 1, linewidth=0, color='green'
                )
            )

# offset y-axis ticks to center them
ax.set_yticks(np.arange(len(ordered_colnames))+0.5)
# need to reverse the order of the labels this time
ax.set_yticklabels(ordered_colnames[::-1], rotation=0)

ax = make_axes_locatable(ax)
# add clinical annotations
ax_sex = ax.append_axes("top", size="7%", pad="2%")
ax_age = ax.append_axes("top", size="7%", pad="2%")
ax_iss = ax.append_axes("top", size="7%", pad="2%")

# Define a colormap for the ISS values
# Define a colormap for the ISS values, including NaN as grey
iss_cmap = ListedColormap(['white', 'orange', 'red', 'grey'])
age_cmap = ListedColormap(['white', 'brown'])
sex_cmap = ListedColormap(['blue', 'pink', 'grey'])


# Plot the 1D heatmap for ISS
iss_values = ordered_clin.loc[ordered_clin.index, 'ISS'].values
age_values = ordered_clin.loc[ordered_clin.index, 'OLD'].values
sex_values = ordered_clin.loc[ordered_clin.index, 'SEX'].values

ax_sex.imshow(sex_values[np.newaxis, :], aspect='auto', cmap=sex_cmap)
ax_age.imshow(age_values[np.newaxis, :], aspect='auto', cmap=age_cmap)
ax_iss.imshow(iss_values[np.newaxis, :], aspect='auto', cmap=iss_cmap)

# Remove ticks
ax_iss.set_xticks([])
ax_iss.set_yticks([])
ax_age.set_xticks([])
ax_age.set_yticks([])
ax_sex.set_xticks([])
ax_sex.set_yticks([])

# Add y-axis labels for ax_iss, ax_age, and ax_sex
ax_sex.set_ylabel('Sex', rotation=0, labelpad=5, va='center', ha='right')
ax_age.set_ylabel('Age', rotation=0, labelpad=5, va='center', ha='right')
ax_iss.set_ylabel('ISS', rotation=0, labelpad=5, va='center', ha='right')

# Add legends for ISS and Age

# Create legend for ISS
iss_legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='I'),
    Patch(facecolor='orange', edgecolor='black', label='II'),
    Patch(facecolor='red', edgecolor='black', label='III'),
    Patch(facecolor='grey', edgecolor='black', label='NA'),
]
ax_iss.legend(handles=iss_legend_elements, title='ISS', bbox_to_anchor=(1, 3), loc='upper left', frameon=False)

# Create legend for Age
age_legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='<63'),
    Patch(facecolor='brown', edgecolor='black', label='≥63'),
]
ax_age.legend(handles=age_legend_elements, title='Age', bbox_to_anchor=(1, -4.5), loc='upper left', frameon=False)

# Create legend for Sex
sex_legend_elements = [
    Patch(facecolor='pink', edgecolor='black', label='F'),
    Patch(facecolor='blue', edgecolor='black', label='M'),
    Patch(facecolor='grey', edgecolor='black', label='NA'),
]
ax_sex.legend(handles=sex_legend_elements, title='Sex', bbox_to_anchor=(1, -9), loc='upper left', frameon=False)


plt.title('RNA-Seq based IgH translocation partner')

plt.gcf().set_size_inches(8, 3)
plt.savefig('assets/heatmap_igh_sides_medium.png', dpi=150, bbox_inches='tight')

# plt.show()
