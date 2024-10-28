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
    
strat = pd.read_csv(os.environ.get("CLINDATAFILE"), sep='\t')\
    .assign(ISS = lambda df: df['D_PT_iss'].astype('category'))\
    [['PUBLIC_ID','ISS']]
    
kmdata = surv.merge(strat, on='PUBLIC_ID')
kmdata = kmdata[(kmdata['pfscdy'] < 3000) & (kmdata['oscdy'] < 3000)]


iss1 = kmdata['ISS'] == 1
iss2 = kmdata['ISS'] == 2
iss3 = kmdata['ISS'] == 3

model_iss1 = KaplanMeierFitter()
model_iss2 = KaplanMeierFitter()
model_iss3 = KaplanMeierFitter()

plt.clf()
ax = plt.subplot(111)

# change to os or pfs accordingly
ax = model_iss1.fit(kmdata.loc[iss1,'oscdy'],kmdata.loc[iss1,'censos'],label='ISS 1').plot_survival_function(ax=ax)

ax = model_iss2.fit(kmdata.loc[iss2,'oscdy'],kmdata.loc[iss2,'censos'],label='ISS 2').plot_survival_function(ax=ax)

ax = model_iss3.fit(kmdata.loc[iss3,'oscdy'],kmdata.loc[iss3,'censos'],label='ISS 3').plot_survival_function(ax=ax)


add_at_risk_counts(model_iss1, model_iss2, model_iss3, ax=ax)
plt.title('Overall survival by ISS stage')
plt.tight_layout()
plt.gcf().set_size_inches(6.5, 5.75)
plt.savefig('assets/km_curve_os_iss.png',dpi=300)