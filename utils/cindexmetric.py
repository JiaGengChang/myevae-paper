import numpy as np
from sksurv.metrics import concordance_index_censored


def ConcordanceIndex(event_indicator, event_time, estimate):
    event_indicator_np = event_indicator.detach().cpu().numpy().astype(bool)
    event_time_np = event_time.detach().cpu().numpy()
    estimate_np = estimate.reshape(-1).detach().cpu().numpy()
    return concordance_index_censored(event_indicator_np, event_time_np, estimate_np)[0]
