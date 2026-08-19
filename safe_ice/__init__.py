"""Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations."""

from importlib.metadata import PackageNotFoundError, version

from .analysis import AdvancedAnalysis, PerformanceEvaluator
from .core import (
    AdaptiveSafeICE,
    ICEvMFNM,
    OptimizedSafeICE,
    SafeICE,
    SubsetSimulation,
    vMFNMParameters,
)
from .distributions import (
    InverseNakagamiDistribution,
    NakagamiDistribution,
    VonMisesFisherSampler,
    vMFNMDistribution,
)
from .optimization import PenalizedEMOptimizer
from .problems import (
    BenchmarkProblems,
    HeatTransferProblem,
    NetworkReliabilityProblem,
    StochasticProcessProblem,
    SystemReliabilityProblem,
    TimeVariantProblem,
)

try:
    __version__ = version("safe-ice")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__author__ = "Diogo Ribeiro"
__all__ = [
    "AdaptiveSafeICE",
    "AdvancedAnalysis",
    "BenchmarkProblems",
    "HeatTransferProblem",
    "ICEvMFNM",
    "InverseNakagamiDistribution",
    "NakagamiDistribution",
    "NetworkReliabilityProblem",
    "OptimizedSafeICE",
    "PenalizedEMOptimizer",
    "PerformanceEvaluator",
    "SafeICE",
    "StochasticProcessProblem",
    "SubsetSimulation",
    "SystemReliabilityProblem",
    "TimeVariantProblem",
    "VonMisesFisherSampler",
    "vMFNMDistribution",
    "vMFNMParameters",
]
