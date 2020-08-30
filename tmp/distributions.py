import numpy as np
import pymc3 as pm
import theano.tensor as tt

__all__ = ["NormalMixture"]

class NormalMixture(pm.distributions.Continuous):
    """ A Gaussian mixture distribution that tends to 
        run a bit faster than the pymc3 version. 
        
        Args:
            w: an array of weights for each component normal
            mu: an array of means for each normal
            sd: an array of standard distributions for each normal
    
    """
    
    def __init__(self, w, mu, sd, *args, **kwargs):
        
        self.size = 1
        self.w = tt.as_tensor_variable(w)
        self.mu = tt.as_tensor_variable(mu)
        self.sd = tt.as_tensor_variable(sd)
        if not "testval" in kwargs:
            kwargs["testval"] = self.mu[tt.argmax(self.w)]
        super(NormalMixture, self).__init__(*args, shape=self.size, **kwargs)
        
    def random(self, point=None, size=(1,)):
        rs = tt.shared_randomstreams.RandomStreams()
        i = rs.choice(a=tt.arange(self.w.size), p=self.w, size=size)
        return rs.normal(self.mu[i], self.sd[i])
    
    def logp(self, x):
        facts = tt.exp(-0.5 * (self.mu[:, None] - x) ** 2 / self.sd[:, None] ** 2)
        coeffs = self.w[:, None] / self.sd[:, None] / tt.sqrt(2*np.pi)
        return tt.log(tt.sum(coeffs * facts, axis=0))