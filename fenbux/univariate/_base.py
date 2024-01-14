from ..core import AbstractDistribution


class UnivariateDistribution(AbstractDistribution):
    pass


class ContinuousUnivariateDistribution(UnivariateDistribution):
    pass


class DiscreteUnivariateDistribution(UnivariateDistribution):
    pass

class TransformedDistribution(UnivariateDistribution):
    pass
