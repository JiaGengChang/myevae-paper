from torch import no_grad, tensor as torch_tensor, cat as torch_cat, load as torch_load
from torch.nn import Module
from sksurv.metrics import concordance_index_censored
import os
import sys
from dotenv import load_dotenv
load_dotenv("../.env")
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.parsers_external import *
from utils.scaler_external import scale_and_impute_external_dataset as scale_impute

def score_external_datasets(model:Module,params:dict,level:str="affy")->tuple[float]:
    """
    Scores a gene expression model on 4 microarray datasets
    1. GSE24080 UAMS
    2. GSE19784 GMMG-HD4/HOVON65
    3. E-MTAB-4032
    4. GSE9782 APEX trial 039
    @Inputs:
        model: an instance of modules_vae.model.MultimodalVAE or modules_deepsurv.Deepsurv
        params: an instance of utils.params.Params
        level: "affy", "entrez", or "ensg". Should use affymetrix probeset as features.
        genes: a list of expression genes to subset the datasets to. Corresponds to the genes the model has seen.
    @Outputs:
        a tuple of C-indexes of the model evaluated on the UAMS, HOVON, and EMTAB microarray datasets
    """
    # the only permitted input types for external validation
    assert params.input_types_all == ['exp', 'clin']

    # subset external validation data to genes seen by the model
    uams_clin_tensor = torch_tensor(scale_impute(parse_clin_uams(), params.scale_method).values)
    uams_exp_tensor = torch_tensor(scale_impute(parse_exp_uams(params.genes,level), params.scale_method).values)

    hovon_clin_tensor = torch_tensor(scale_impute(parse_clin_hovon(), params.scale_method).values)
    hovon_exp_tensor = torch_tensor(scale_impute(parse_exp_hovon(params.genes,level), params.scale_method).values)

    emtab_clin_tensor = torch_tensor(scale_impute(parse_clin_emtab(), params.scale_method).values)
    emtab_exp_tensor = torch_tensor(scale_impute(parse_exp_emtab(params.genes,level), params.scale_method).values)

    apex_clin_tensor = torch_tensor(scale_impute(parse_clin_apex(),params.scale_method).values)
    apex_exp_tensor = torch_tensor(scale_impute(parse_exp_apex(params.genes,level), params.scale_method).values)
    
    uams_events, uams_times = parse_surv_uams(params.endpoint)
    hovon_events, hovon_times = parse_surv_hovon(params.endpoint)
    emtab_events, emtab_times = parse_surv_emtab(params.endpoint)
    apex_events, apex_times = parse_surv_apex(params.endpoint)

    if params.architecture=="VAE":
        # use the VAE API for scoring. VAE is a torch.nn.Module
        model.eval()
        with no_grad():
            _, _, _, estimates_uams =  model([[uams_exp_tensor], [uams_clin_tensor]])
            _, _, _, estimates_hovon = model([[hovon_exp_tensor], [hovon_clin_tensor]])
            _, _, _, estimates_emtab = model([[emtab_exp_tensor], [emtab_clin_tensor]])
            _, _, _, estimates_apex = model([[apex_exp_tensor], [apex_clin_tensor]])
    elif params.architecture=='Deepsurv':
        model.eval()
        # use the Deepsurv API for scoring. model is a CoxPH object (pycox)
        with no_grad():
            # potential bug, I don't know if -1 is the concatenation axis
            estimates_uams =  model(torch_cat([uams_exp_tensor, uams_clin_tensor],axis=-1))
            estimates_hovon = model(torch_cat([hovon_exp_tensor, hovon_clin_tensor],axis=-1))
            estimates_emtab = model(torch_cat([emtab_exp_tensor, emtab_clin_tensor],axis=-1))
            estimates_apex = model(torch_cat([apex_exp_tensor,apex_clin_tensor],axis=-1))
    else:
        raise NotImplementedError(params.architecture)

    cindex_uams = concordance_index_censored(uams_events, uams_times, estimates_uams.flatten())[0].item()
    cindex_hovon = concordance_index_censored(hovon_events, hovon_times, estimates_hovon.flatten())[0].item()
    cindex_emtab = concordance_index_censored(emtab_events, emtab_times, estimates_emtab.flatten())[0].item()
    cindex_apex = concordance_index_censored(apex_events, apex_times, estimates_apex.flatten())[0].item()

    return max(cindex_uams,1-cindex_uams), max(cindex_hovon,1-cindex_hovon), max(cindex_emtab,1-cindex_emtab), max(cindex_apex,1-cindex_apex)   

