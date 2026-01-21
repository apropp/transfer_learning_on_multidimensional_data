"""
utils_UQ.py: This script defines helper functions for UQ experiments
             used to evaluate our CNN surrogate.

Title:  TRANSFER LEARNING ON MULTI-DIMENSIONAL DATA: 
        A NOVEL APPROACH TO NEURAL NETWORK-BASED SURROGATE MODELING
        DOI: 10.1615/JMachLearnModelComput.2024057138

author: A.M. Propp
email: propp@stanford.edu
GitHub: apropp
Updated: May 2024
"""

import numpy as np

def bt_cdf_pdf_MC(dat, qoi, dps=False, threshold=0.15, dist_breakthru=100, dist_max=150):
    # ns = number of samples
    # dps = number of samples
    # qoi would be, e.g. 'ss_hfs'
    if dps is False:
        ns = dat[qoi].shape[0] # all samples in data file
        idx_sample = np.arange(ns)
    else:
        ns = dps
        idx_sample = np.random.choice(np.arange(dat[qoi].shape[0]),size=ns,replace=False) # random sample
    ts = dat[qoi].shape[1] # number of timesteps
    nx = dat[qoi].shape[3] # domain size?
    lastcol = int(dist_breakthru/dist_max*nx)
    pdf_count = [0] * (ts+1)
    cdf_count = [0] * (ts+1)
    # default is last timestep
    out = [17] * ns
    # iterate through all the samples?
    for i in range(ns):
        t = 0
        while out[i] == 17 and t<ts:
            max_lastcol = max(dat[qoi][idx_sample[i],t,:,lastcol-1])
            max_before_1 = max(dat[qoi][idx_sample[i],t,:,lastcol-2])
            max_before_2 = max(dat[qoi][idx_sample[i],t,:,lastcol-3])
            max_before_3 = max(dat[qoi][idx_sample[i],t,:,lastcol-4])
            # 4 consecutive columns need to be above threshold
            if max_lastcol > threshold and max_before_1 > threshold and max_before_2 > threshold and max_before_3 > threshold:
                out[i] = t+1
            t+=1
        # build PDF (increment count for the timestep at which we surpassed threshold at dist_breakthrough)
        pdf_count[(out[i]-1)] += 1

    # calculate CDF
    cdf_sum = 0
    for i,value in enumerate(pdf_count):
        cdf_sum += value
        cdf_count[i] = cdf_sum

    cdf = np.array(cdf_count) / ns
    pdf = np.array(pdf_count) / ns
    return cdf, pdf


def bt_cdf_pdf_TL(model_out, dps=False, threshold=0.1, dist_breakthru=150, dist_max=150):
    if dps is False:
        ns = model_out.shape[0]
        idx_sample = np.arange(ns)
    else:
        ns = dps
        idx_sample = np.random.choice(np.arange(model_out.shape[0]),size=ns,replace=False)
    ts = model_out.shape[1]
    nx = model_out.shape[3]
    lastcol = int(dist_breakthru/dist_max*nx)
    pdf_count = [0] * (ts+1)
    cdf_count = [0] * (ts+1)
    out = [17] * ns
    for i in range(ns):
        t = 0
        while out[i] == 17 and t<ts:
            max_lastcol = max(model_out[idx_sample[i],t,:,lastcol-1])
            max_before_1 = max(model_out[idx_sample[i],t,:,lastcol-2])
            max_before_2 = max(model_out[idx_sample[i],t,:,lastcol-3])
            max_before_3 = max(model_out[idx_sample[i],t,:,lastcol-4])
            if max_lastcol > threshold and max_before_1 > threshold and max_before_2 > threshold and max_before_3 > threshold:
                out[i] = t+1
            t+=1
        pdf_count[(out[i]-1)] += 1

    # calculate PDF
    cdf_sum = 0
    for i,value in enumerate(pdf_count):
        cdf_sum += value
        cdf_count[i] = cdf_sum

    cdf = np.array(cdf_count) / ns
    pdf = np.array(pdf_count) / ns
    return cdf, pdf


def kl_divergence(p_org,q_org,sigma=0):
    if sigma != 0:
        p = gaussian_filter(p_org,sigma)
        q = gaussian_filter(q_org,sigma)
    else:
        p = p_org
        q = q_org
    p = p/(np.sum(p))
    q = q/(np.sum(q))
    if any(p==0.0):
        print('KL 0 issue')
    #p[p==0.0]=1e-10
    #q[q==0.0]=1e-10
    return np.sum(p * np.log(p/q) )
