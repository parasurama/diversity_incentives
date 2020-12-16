#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Author: Prasanna Parasurama
"""

import numpy as np
from numba import njit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import beta
import scipy.stats as st

sns.set_style("ticks")
T_TRIALS = 100000


@njit()
def generate_apps(n_males, n_females):
    q_m = np.random.uniform(0, 1, n_males)  # quality of male apps
    q_f = np.random.uniform(0, 1, n_females)  # quality of female apps
    return q_m, q_f


@njit()
def shortlist(q_m, q_f, b):
    u_m = q_m  # utility if hired
    u_f = b*q_f  # utility if hired

    qs = np.append(q_m, q_f)
    us = np.append(u_m, u_f)

    top_idx = np.argmax(us)  # idx of max utility
    q = qs[top_idx]  # quality of shortlisted candidate
    u = us[top_idx]  # utility from shortlisted candidate
    is_male_shortlisted = int(top_idx < len(q_m)) # gender of shortlisted candidate
    return np.array([is_male_shortlisted, q, u])  # male, quality, utility


@njit()
def simulate_shortlist(n_males, n_females, b):
    results = np.empty(shape=(T_TRIALS, 3))
    for t in range(T_TRIALS):
        q_m, q_f = generate_apps(n_males, n_females)
        s = shortlist(q_m=q_m, q_f=q_f, b=b)
        results[t] = s

    # print("E[male] = ", results[:, 0].mean())
    # print("E[quality] = ", results[:, 1].mean())
    # print("E[util] = ", results[:, 2].mean())
    return results  # [[male, quality, utility],[...]]


@njit()
def hire(n_males, n_females, n_shortlist, b):
    shortlisted_candidates = np.zeros(shape=(n_shortlist, 3))
    for r in range(n_shortlist):
        q_m, q_f = generate_apps(n_males, n_females)
        shortlisted_candidates[r] = shortlist(q_m, q_f, b)

    idx_max_q = np.argmax(shortlisted_candidates[:, 1])  # idx of highest q

    is_male_hired = shortlisted_candidates[idx_max_q, 0]
    q_hired = shortlisted_candidates[idx_max_q, 1]  # quality of highest q
    n_males_shortlisted = shortlisted_candidates[:, 0].sum()
    return np.array([is_male_hired, q_hired, n_males_shortlisted])


@njit()
def simulate_hiring(n_males, n_females, n_shortlist, b):
    results = np.empty(shape=(T_TRIALS, 3))
    for t in range(T_TRIALS):
        h = hire(n_males, n_females, n_shortlist, b)
        results[t] = h
    return results


def get_hiring_simulation_metrics(n_males, n_females, n_to_be_shortlisted, b):
    sim_results = simulate_hiring(n_males, n_females, n_to_be_shortlisted, b)
    dfs = pd.DataFrame(sim_results, columns=['is_male_hired', 'q_hire', 'n_males_shortlisted'])
    dfs['n_shortlisted'] = n_to_be_shortlisted
    dfs['n_females_shortlisted'] = dfs['n_shortlisted'] - dfs['n_males_shortlisted']
    dfs['n_males_applied'] = n_males*n_to_be_shortlisted
    return dfs


# theoretical predictions

def prob_male_shortlisted(nm, nf, b):
    """
    theoretical prob that male is hired
    :param nm:
    :type nm:
    :param nf:
    :type nf:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return (b ** (-nf) * nm)/(nf+nm)


def cond_prob_male_hired(nm, nf, b):
    """
    conditional prob that male is hired given a shortlist of male, female
    :param nm:
    :type nm:
    :param nf:
    :type nf:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return (b ** (-nf - nm) * (nf * nm + 2 * b ** (2 * nf + nm) * (nf + nm) ** 2 - 2 * b ** (nf + nm) * nm * (2 * nf + nm))) / (2. * (2 * nf + nm) * (-nm + b ** nf * (nf + nm)))


def total_probability_male_hired(nm, nf, b):
    pr_male_sl = prob_male_shortlisted(nm, nf, b)

    pr_n_male_sl = st.binom(n=2, p=pr_male_sl)

    pr_zero_male_sl = pr_n_male_sl.pmf(0)
    pr_one_male_sl = pr_n_male_sl.pmf(1)
    pr_two_male_sl = pr_n_male_sl.pmf(2)

    pred_prob_male_hired = pr_zero_male_sl * 0 + pr_one_male_sl * cond_prob_male_hired(nm, nf, b) + pr_two_male_sl * 1
    return pred_prob_male_hired


def pdfUtilShortlistF(x, nm, nf, b):
    if x >=1:
        d = (nf * (nf + nm) * x**(-1+nf))/(-nm + b**nf * (nf + nm))
    else:
        d = (nf * (nf + nm) * x**(-1+nf) * x**nm) / (-nm + b**nf * (nf + nm))
    return d

@njit()
def pdfQualShortlistF(x, nm, nf, b):
    if 0 <= x < 1/b:
        d = (b*nf * (nf + nm) * (b*x)**(-1+nf) * (b*x)**nm)/(-nm + b**nf * (nf + nm))
    elif 1/b <= x <= 1:
        d = (b*nf * (nf + nm) * (b*x)**(-1+nf))/(-nm + b**nf * (nf + nm))
    else:
        d = 0
    return d

def pdfUtilShortlistM(x, nm, nf, b):
    if x > 1:
        return 0
    else:
        return (nf+nm) * x**(-1+nf+nm)/(nm * beta(nm, 1))

class RVQualSLF(st.rv_continuous):
    def _pdf(self, x):
        return pdfQualShortlistF(x, nm=N_M, nf=N_F, b=B)

class RVQualSLM(st.rv_continuous):
    def _pdf(self, x):
        return pdfUtilShortlistM(x, nm=N_M, nf=N_F, b=B)


def expUtilShortlistF(nm, nf, b):
    """
    expected utility of a shortlisted female
    :param nm:
    :type nm:
    :param nf:
    :type nf:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return (nf * (nf + nm) * (-nm + b ** (1 + nf) * (1 + nf + nm))) / ((1 + nf) * (1 + nf + nm) * (-nm + b ** nf * (nf + nm)))


def expQualShortlistF(nm, nf, b):
    """
    expected quality of shortlisted female
    :param nm:
    :type nm:
    :param nf:
    :type nf:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return expUtilShortlistF(nm, nf, b)/b

def expQualShortlistM(nm, nf, b):
    """
    expected quality of shortlisted male
    :param nm:
    :type nm:
    :param nf:
    :type nf:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return (nf+nm)/(1+nf+nm)


# compare simulations and predictions
N_M = 3
N_F = 1
B = 1.6

# generate data
df = pd.DataFrame(simulate_shortlist(n_males=N_M, n_females=N_F, b=B), columns=['m', 'q', 'u'])
dfh = get_hiring_simulation_metrics(N_M, N_F, 2, B)

dfp = pd.DataFrame({'x': np.linspace(0, 1.5, 10000)})
dfp['pdfUtilShortlistF'] = dfp['x'].apply(lambda x: pdfUtilShortlistF(x=x, nm=N_M, nf=N_F, b=B))
dfp['pdfQualShortlistF'] = dfp['x'].apply(lambda x: pdfQualShortlistF(x=x, nm=N_M, nf=N_F, b=B))
dfp['pdfQualShortlistM'] = dfp['x'].apply(lambda x: pdfUtilShortlistM(x=x, nm=N_M, nf=N_F, b=B))

female_qual_rvs = RVQualSLF(a=0, b=1).rvs(size=10000)
male_qual_rvs = RVQualSLM(a=0, b=1).rvs(size=10000)



# probability of male is shortlisted
print(df['m'].mean())
print(prob_male_shortlisted(N_M, N_F, B))

# probability of shortlist as a function of bonus
dfb = pd.DataFrame({'b': np.linspace(1,3, 1000)})
dfb['male'] = dfb['b'].apply(lambda x: prob_male_shortlisted(nm=N_M, nf=N_F, b=x))
dfb['female'] = 1-dfb['male']

sns.lineplot(x='b', y='Probability of being shortlisted', hue='gender',
             data=dfb.melt('b', value_name="Probability of being shortlisted", var_name="gender"))
sns.despine()
plt.show()


# expected util of shortlisted female
print(df[df['m'] == 0]['u'].mean())
print(expUtilShortlistF(N_M, N_F, B))

# expected qual of shortlisted female
print(df[df['m'] == 0]['q'].mean())
print(expQualShortlistF(N_M, N_F, B))

# expected qual of shortlisted male
print(df[df['m'] == 1]['q'].mean())
print(expQualShortlistM(N_M, N_F, B))

# expected quality as a function of bonus
dfb = pd.DataFrame({'b': np.linspace(1,3, 1000)})
dfb['male'] = dfb['b'].apply(lambda x: expQualShortlistM(N_M, N_F, x))
dfb['female'] = dfb['b'].apply(lambda x: expQualShortlistF(N_M, N_F, x))

sns.lineplot(x='b', y='Expected quality of shortlisted candidate', hue='gender',
             data=dfb.melt('b', value_name="Expected quality of shortlisted candidate", var_name="gender"))
sns.despine()
plt.show()

# density of util of shortlisted female
sns.distplot(df[df['m'] == 0]['u'])
sns.lineplot(x='x', y='pdfUtilShortlistF', data=dfp)
plt.show()

# density of qual of shortlisted female
sns.distplot(df[df['m'] == 0]['q'])
sns.lineplot(x='x', y='pdfQualShortlistF', data=dfp)


sns.distplot(female_qual_rvs)
plt.show()


# density of qual of shortlisted male
sns.distplot(df[df['m'] == 1]['q'])
sns.lineplot(x='x', y='pdfQualShortlistM', data=dfp[dfp['x'] < 1])
sns.distplot(male_qual_rvs)
plt.show()

# compare density of qual of shortlisted male and female
dfpq = dfp.melt('x')
dfpq = dfpq[dfpq['variable'].isin(["pdfQualShortlistM", "pdfQualShortlistF"])]
dfpq = dfpq[dfpq['x'] < 1]
sns.lineplot(x='x', y='value', hue='variable', data=dfpq)
plt.show()


# conditional prob of male hire
dfb = pd.DataFrame({'b': np.linspace(1,3, 1000)})
dfb['male'] = dfb['b'].apply(lambda x: cond_prob_male_hired(N_M, N_F, x))
dfb['female'] = 1-dfb['male']

sns.lineplot(x='b', y='Conditional probability of being hired', hue='gender',
             data=dfb.melt('b', value_name="Conditional probability of being hired", var_name="gender"))
sns.despine()
plt.show()

# prob of male hired
print(total_probability_male_hired(N_M,N_F,B))
print(dfh['is_male_hired'].mean())

dfb = pd.DataFrame({'b': np.linspace(1,3, 1000)})
dfb['male'] = dfb['b'].apply(lambda x: total_probability_male_hired(N_M, N_F, x))
dfb['female'] = 1-dfb['male']

sns.lineplot(x='b', y='Overall probability of being hired', hue='gender',
             data=dfb.melt('b', value_name="Overall probability of being hired", var_name="gender"))
sns.despine()
plt.show()


