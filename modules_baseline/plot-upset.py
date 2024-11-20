import sys
sys.path.append('utils/')
from parsers import *
from upsetplot import plot, from_contents
from matplotlib import pyplot as plt

ovs = parse_surv('os').dropna().PUBLIC_ID
pfs = parse_surv('pfs').dropna().PUBLIC_ID
rna = parse_rna().dropna().PUBLIC_ID
cna = parse_cna().dropna().PUBLIC_ID
sbs = parse_sbs().dropna().PUBLIC_ID
igh = parse_sv().dropna().PUBLIC_ID
clin = parse_clin().dropna().PUBLIC_ID
fish = parse_fish().dropna().PUBLIC_ID
gistic = parse_gistic().dropna().PUBLIC_ID
chromoth = parse_chromoth().dropna().PUBLIC_ID

data = from_contents({
    "Overall survival": ovs,
    "Progression-free survival": pfs,
    "Clinical (age,sex,ISS)": clin,
    # "WGS-based iFISH": fish, # same as CNA
    # "WGS GISTIC peaks": gistic, # same as CNA
    "WES SBS MutSig": sbs,
    "WGS Copy num.": cna,
    "WGS Chromothripsis.": chromoth,
    "RNA-Seq IG subtypes": igh,
    "RNA-Seq Gene Expr": rna,
})
plt.tight_layout()
plt.gcf().set_size_inches(5, 5)

plot(data,show_counts=True,min_subset_size=10)
plt.savefig('assets/upsetplot.png',dpi=300)

# plot(data,show_counts=True,min_subset_size=1)
# plt.savefig('assets/upsetplot_full.png',dpi=300)