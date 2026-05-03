from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
gmm.fit(glitch_size_data6910.reshape(-1, 1))

means = gmm.means_.ravel()
stds = np.sqrt(gmm.covariances_.ravel())

print('means:', means, 'stds:', stds)
