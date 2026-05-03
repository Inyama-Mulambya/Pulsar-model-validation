#Define log-likelihood functions
def log_likelihood_lognormal(params, data):
    mean, sigma = params
    return np.sum(lognorm.logpdf(data, sigma, scale=np.exp(mean)))

def log_likelihood_exponential(params, data):
    decay = params[0]
    return np.sum(np.log(decay * np.exp(-decay * data)))

def log_likelihood_gaussian(params, data):
    mean, sigma = params
    return np.sum(norm.logpdf(data, mean, sigma))

def log_likelihood_powerlaw(params, data):
    alpha, xmin = params
    return np.sum(np.log((alpha - 1) / xmin * (data / xmin)**(-alpha)))

def log_likelihood_lognormal_lognormal(params, data):
    amp1, mean1, sigma1, amp2, mean2, sigma2 = params
    ll1 = amp1 * lognorm.pdf(data, sigma1, scale=np.exp(mean1))
    ll2 = amp2 * lognorm.pdf(data, sigma2, scale=np.exp(mean2))
    return np.sum(np.log(ll1 + ll2))

def log_likelihood_gaussian_gaussian(params, data):
    amp1, mean1, sigma1, amp2, mean2, sigma2 = params
    ll1 = amp1 * norm.pdf(data, mean1, sigma1)
    ll2 = amp2 * norm.pdf(data, mean2, sigma2)
    return np.sum(np.log(ll1 + ll2))

def log_likelihood_lognormal_gaussian(params, data):
    amp1, mean1, sigma1, amp2, mean2, sigma2 = params
    ll1 = amp1 * lognorm.pdf(data, sigma1, scale=np.exp(mean1))
    ll2 = amp2 * norm.pdf(data, mean2, sigma2)
    return np.sum(np.log(ll1 + ll2))

#Define parameters and perform fits
distributions = {
    'lognormal': {'func': log_likelihood_lognormal, 'params': np.array([np.mean(np.log(data)), np.std(np.log(data))])},
    'exponential': {'func': log_likelihood_exponential, 'params': np.array([0.01])},
    'gaussian': {'func': log_likelihood_gaussian, 'params': np.array([np.mean(data), np.std(data)])},
    'powerlaw': {'func': log_likelihood_powerlaw, 'params': np.array([2, np.min(data)])},
    'lognormal_lognormal': {'func': log_likelihood_lognormal_lognormal, 'params': np.array([0.5, np.mean(np.log(data)), np.std(np.log(data)), 0.5, np.mean(np.log(data)), np.std(np.log(data))])},
    'gaussian_gaussian': {'func': log_likelihood_gaussian_gaussian, 'params': np.array([0.5, np.mean(data), np.std(data), 0.5, np.mean(data), np.std(data)])},
    'lognormal_gaussian': {'func': log_likelihood_lognormal_gaussian, 'params': np.array([0.5, np.mean(np.log(data)), np.std(np.log(data)), 0.5, np.mean(data), np.std(data)])},
}

#Perform fits and calculate BIC values
bic_values = {}
for dist_name, dist_info in distributions.items():
    result = minimize(lambda params: -dist_info['func'](params, data), dist_info['params'], method="SLSQP")
    bic = -2 * dist_info['func'](result.x, data) + len(dist_info['params']) * np.log(len(data))
    bic_values[dist_name] = bic

#Print BIC values
for dist_name, bic in bic_values.items():
    print(f"BIC for {dist_name}: {bic}")

#Determine the best model
best_model = min(bic_values, key=bic_values.get)
print(f"Best model: {best_model}")
