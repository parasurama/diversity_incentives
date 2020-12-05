#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Author: Prasanna Parasurama
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
gpa = np.random.uniform(0, 10, size=1000)
yrs_exp = np.random.uniform(0, 10, size=1000)
epsilon = np.random.uniform(0,1,size=10000)

sns.histplot(gpa+yrs_exp)
plt.show()


def logit(x):
    return np.log(x/(1-x))

q_hat = np.random.uniform(0,1,10000)
e = 1/(1+np.exp(-np.random.normal(logit(q_hat), 1)))
q = 0.7*q_hat + 0.3*e

sns.histplot(q_hat)
plt.show()

sns.histplot(q)
plt.show()

df = pd.DataFrame({'q': q, 'q_hat': q_hat})
sns.lmplot(x='q', y='q_hat', data=df)
plt.show()