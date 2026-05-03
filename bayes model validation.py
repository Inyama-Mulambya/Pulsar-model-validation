#defining bayes double lognormal distribution model
with pm.Model() as model:
  mu_prior1 = pm.Normal('mu_prior1', mu = 141.75131935, sigma = 88.39169691)
  sigma_prior1 = pm.HalfNormal('sigma_prior1', sigma = 88.39169691)
  mu_prior2 = pm.Normal('mu_prior2', mu = 402.56593194, sigma = 106.75426626)
  sigma_prior2 = pm.HalfNormal('sigma_prior2', sigma = 106.75426626)
  w = pm.Dirichlet('w', np.ones(2))
  dist1 = pm.Lognormal.dist(mu= mu_prior1, sigma =sigma_prior1)
  dist2 = pm.Lognormal.dist(mu= mu_prior2, sigma=sigma_prior2)
  #bimodal=pm.Mixture('bimodal', w=w, comp_dists=[dist1, dist2])
  bimodal_obs = pm.Mixture('bimodal_obs', w=w, comp_dists=[dist1, dist2], observed=glitch_size_data6910)

#perform MCMC simulations
with model:
  trace = pm.sample(20000, tune = 10000, step=pm.NUTS(target_accept=0.95), chains = 4, cores = 4, return_inferencedata=True) # Set return_inferencedata=True to get InferenceData object directly

  # Add new values as new variables to the dataset
  trace.posterior = trace.posterior.assign(
      mu_orig1 = (('chain', 'draw'), np.exp(trace.posterior['mu_prior1']).values),
      sigma_orig1 = (('chain', 'draw'), np.exp(trace.posterior['sigma_prior1']).values),
      mu_orig2 = (('chain', 'draw'), np.exp(trace.posterior['mu_prior2']).values),
      sigma_orig2 = (('chain', 'draw'), np.exp(trace.posterior['sigma_prior2']).values)
  )
