import sys
sys.path.append('utils/')
from parsers import *
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

# KM curve for ISS stage
import os
from dotenv import load_dotenv     
load_dotenv('.env')

surv = pd.read_csv(os.environ.get("SURVDATAFILE"), sep='\t')\
    [['PUBLIC_ID','oscdy','censos','pfscdy','censpfs']]\
    .dropna()
    
data = pd.read_csv(os.environ.get("CLINDATAFILE"), sep='\t')

kmdata = surv.merge(data, on='PUBLIC_ID')

kmdata = kmdata[(kmdata['pfscdy'] < 3000) & (kmdata['oscdy'] < 3000)]

plt.clf()
ax = plt.subplot(111)

# change to os or pfs accordingly
kmf0 = KaplanMeierFitter()
kmf1 = KaplanMeierFitter()
age0 = kmdata['D_PT_age'] < 63
age1 = kmdata['D_PT_age'] >= 63
ax = kmf0.fit(kmdata.loc[age0,'pfscdy'], kmdata.loc[age0,'censpfs'], label='age<63').plot_survival_function(ax=ax)
ax = kmf1.fit(kmdata.loc[age1,'pfscdy'], kmdata.loc[age1,'censpfs'], label='age>=63').plot_survival_function(ax=ax)

add_at_risk_counts(kmf0, kmf1, ax=ax, ypos=-0.3)

plt.title('Progression-free survival by age at diagnosis')
plt.tight_layout()
plt.gcf().set_size_inches(5, 4.25)
plt.savefig('assets/km_curve_pfs_age.png',dpi=300)