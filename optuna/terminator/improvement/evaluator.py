from __future__ import annotations

import abc
import math
import sys
from typing import cast
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import intersection_search_space
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import scipy.stats as scipy_stats
    import torch
    from optuna._gp import acqf
    from optuna._gp import gp
    from optuna._gp import optim_sample
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp = _LazyImport("optuna._gp.gp")
    optim_sample = _LazyImport("optuna._gp.optim_sample")
    acqf = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    scipy_stats = _LazyImport("scipy.stats")

DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


@experimental_class("3.2.0")
class BaseImprovementEvaluator(metaclass=abc.ABCMeta):
    """Base class for improvement evaluators."""

    @abc.abstractmethod
    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        pass


@experimental_class("3.2.0")
class RegretBoundEvaluator(BaseImprovementEvaluator):
    """An error evaluator for upper bound on the regret with high-probability confidence.

    This evaluator evaluates the regret of current best solution, which defined as the difference
    between the objective value of the best solution and of the global optimum. To be specific,
    this evaluator calculates the upper bound on the regret based on the fact that empirical
    estimator of the objective function is bounded by lower and upper confidence bounds with
    high probability under the Gaussian process model assumption.

    Args:
        gp:
            A Gaussian process model on which evaluation base. If not specified, the default
            Gaussian process model is used.
        top_trials_ratio:
            A ratio of top trials to be considered when estimating the regret. Default to 0.5.
        min_n_trials:
            A minimum number of complete trials to estimate the regret. Default to 20.
        min_lcb_n_additional_samples:
            A minimum number of additional samples to estimate the lower confidence bound.
            Default to 2000.

    For further information about this evaluator, please refer to the following paper:

    - `Automatic Termination for Hyperparameter Optimization <https://proceedings.mlr.press/v188/makarova22a.html>`__
    """  # NOQA: E501

    def __init__(
        self,
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        seed: int | None = None,
    ) -> None:
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials
        self._log_prior = prior.default_log_prior
        self._minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
        self._optimize_n_samples = 2048
        self._rng = LazyRandomState(seed)

    def _get_top_n(
        self, normalized_params: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert len(normalized_params) == len(values)
        n_trials = len(normalized_params)
        top_n = np.clip(int(n_trials * self._top_trials_ratio), self._min_n_trials, n_trials)
        top_n_val = np.partition(values, n_trials - top_n)[n_trials - top_n]
        top_n_mask = values >= top_n_val
        return normalized_params[top_n_mask], values[top_n_mask]

    @staticmethod
    def _get_beta(n_params: int, n_trials: int, delta: float = 0.1) -> float:
        # TODO(nabenabe0928): Check the original implementation to verify.
        # Especially, |D| seems to be the domain size, but not the dimension based on Theorem 1.
        beta = 2 * np.log(n_params * n_trials**2 * np.pi**2 / 6 / delta)

        # The following div is according to the original paper: "We then further scale it down
        # by a factor of 5 as defined in the experiments in
        # `Srinivas et al. (2010) <https://dl.acm.org/doi/10.5555/3104322.3104451>`__"
        beta /= 5

        return beta

    @staticmethod
    def _compute_standardized_regret_bound(
        kernel_params: gp.KernelParamsTensor,
        considered_search_space: gp_search_space.SearchSpace,
        normalized_top_n_params: np.ndarray,
        standarized_top_n_values: np.ndarray,
        delta: float = 0.1,
        optimize_n_samples: int = 2048,
        rng: np.random.RandomState | None = None
    ) -> float:
        """
        In the original paper, f(x) was intended to be minimized, but here we would like to
        maximize f(x). Hence, the following changes happen:
            1. min(ucb) over top trials becomes max(lcb) over top trials, and
            2. min(lcb) over the search space becomes max(ucb) over the search space, and
            3. Regret bound becomes max(ucb) over the search space minus max(lcb) over top trials.
        """

        n_trials, n_params = normalized_top_n_params.shape

        # calculate max_ucb
        beta = RegretBoundEvaluator._get_beta(n_params, n_trials, delta)
        ucb_acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.UCB,
            kernel_params=kernel_params,
            search_space=considered_search_space,
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            beta=beta,
        )
        # UCB over the search space. (Original: LCB over the search space. See Change 1 above.)
        standardized_ucb_value = max(
            acqf.eval_acqf_no_grad(ucb_acqf_params, normalized_top_n_params).max(),
            optim_sample.optimize_acqf_sample(
                ucb_acqf_params, n_samples=optimize_n_samples, rng=rng
            )[1],
        )

        # calculate min_lcb
        lcb_acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.LCB,
            kernel_params=kernel_params,
            search_space=considered_search_space,
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            beta=beta,
        )
        # LCB over the top trials. (Original: UCB over the top trials. See Change 2 above.)
        standardized_lcb_value = np.max(
            acqf.eval_acqf_no_grad(lcb_acqf_params, normalized_top_n_params)
        )

        # max(UCB) - max(LCB). (Original: min(UCB) - min(LCB). See Change 3 above.)
        return standardized_ucb_value - standardized_lcb_value  # standardized regret bound

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        optuna_search_space = intersection_search_space(trials)
        self._validate_input(trials, optuna_search_space)

        complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        # _gp module assumes that optimization direction is maximization
        sign = -1 if study_direction == StudyDirection.MINIMIZE else 1
        values = np.array([t.value for t in complete_trials]) * sign
        considered_search_space, normalized_params = (
            gp_search_space.get_search_space_and_normalized_params(
                complete_trials, optuna_search_space
            )
        )
        normalized_top_n_params, top_n_values = self._get_top_n(normalized_params, values)
        top_n_values_mean = top_n_values.mean()
        top_n_values_std = max(1e-10, top_n_values.std())
        standarized_top_n_values = (top_n_values - top_n_values_mean) / top_n_values_std

        kernel_params = gp.fit_kernel_params(
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            is_categorical=(
                considered_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            # TODO(contramundum53): Add option to specify this.
            deterministic_objective=False,
            # TODO(y0z): Add `kernel_params_cache` to speedup.
            initial_kernel_params=None,
        )

        standardized_regret_bound = RegretBoundEvaluator._compute_standardized_regret_bound(
            kernel_params,
            considered_search_space,
            normalized_top_n_params,
            standarized_top_n_values,
            rng=self._rng.rng
        )
        return standardized_regret_bound * top_n_values_std  # regret bound

    @classmethod
    def _validate_input(
        cls, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the regret bound cannot be evaluated."
            )

        if len(search_space) == 0:
            raise ValueError(
                "The intersection search space is empty. This condition is not supported by "
                f"{cls.__name__}."
            )


@experimental_class("3.4.0")
class BestValueStagnationEvaluator(BaseImprovementEvaluator):
    """Evaluates the stagnation period of the best value in an optimization process.

    This class is initialized with a maximum stagnation period (`max_stagnation_trials`)
    and is designed to evaluate the remaining trials before reaching this maximum period
    of allowed stagnation. If this remaining trials reach zero, the trial terminates.
    Therefore, the default error evaluator is instantiated by StaticErrorEvaluator(const=0).

    Args:
        max_stagnation_trials:
            The maximum number of trials allowed for stagnation.
    """

    def __init__(self, max_stagnation_trials: int = 30) -> None:
        if max_stagnation_trials < 0:
            raise ValueError("The maximum number of stagnant trials must not be negative.")
        self._max_stagnation_trials = max_stagnation_trials

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        self._validate_input(trials)
        is_maximize_direction = True if (study_direction == StudyDirection.MAXIMIZE) else False
        trials = [t for t in trials if t.state == TrialState.COMPLETE]
        current_step = len(trials) - 1

        best_step = 0
        for i, trial in enumerate(trials):
            best_value = trials[best_step].value
            current_value = trial.value
            assert best_value is not None
            assert current_value is not None
            if is_maximize_direction and (best_value < current_value):
                best_step = i
            elif (not is_maximize_direction) and (best_value > current_value):
                best_step = i

        return self._max_stagnation_trials - (current_step - best_step)

    @classmethod
    def _validate_input(cls, trials: list[FrozenTrial]) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the improvement cannot be evaluated."
            )


@experimental_class("4.0.0")
class EMMREvaluator(BaseImprovementEvaluator):
    """Evaluates a kind of regrets, called the Expected Minimum Model Regret(EMMR).

    EMMR is an upper bound of "expected minimum simple regret" in the optimization process.

    Expected minimum simple regret is a quantity that converges to zero only if the
    optimization process has found the global optima.

    For further information about expected minimum simple regret and the algorithm,
    please refer to the following paper:

    - `A stopping criterion for Bayesian optimization by the gap of expected minimum simple
      regrets <https://proceedings.mlr.press/v206/ishibashi23a.html>`__

    Args:
    """

    def __init__(
        self,
        deterministic_objective: bool = False,
        delta: float = 0.1,
        min_n_trials: int = 20,
        seed: int | None = None,
    ) -> None:
        if min_n_trials <= 1 or not np.isfinite(min_n_trials):
            raise ValueError("`min_n_trials` is expected to be an integer more than two.")

        self._deterministic = deterministic_objective
        self._delta = delta
        self.min_n_trials = min_n_trials
        self._rng = LazyRandomState(seed)

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:

        optuna_search_space = intersection_search_space(trials)
        complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        if len(complete_trials) < 3:
            return sys.float_info.max  # Do not terminate.

        considered_search_space, normalized_params = (
            gp_search_space.get_search_space_and_normalized_params(
                complete_trials, optuna_search_space
            )
        )
        if len(considered_search_space.scale_types) == 0:
            return sys.float_info.max  # Do not terminate.

        len_trials = len(complete_trials)
        len_params = len(considered_search_space.scale_types)
        assert normalized_params.shape == (len_trials, len_params)

        # _gp module assumes that optimization direction is maximization
        sign = -1 if study_direction == StudyDirection.MINIMIZE else 1
        score_vals = np.array([cast(float, t.value) for t in complete_trials]) * sign

        if np.any(~np.isfinite(score_vals)):
            warnings.warn(
                f"{self.__class__.__name__} cannot handle infinite values."
                "Those values are clamped to worst/best finite value."
            )

            finite_score_vals = score_vals[np.isfinite(score_vals)]
            best_finite_score = np.max(finite_score_vals, initial=0.0)
            worst_finite_score = np.min(finite_score_vals, initial=0.0)

            score_vals = np.clip(score_vals, worst_finite_score, best_finite_score)

        standarized_score_vals = (score_vals - score_vals.mean()) / max(
            sys.float_info.min, score_vals.std()
        )

        assert len(standarized_score_vals) == len(normalized_params)

        kernel_params_t2 = gp.fit_kernel_params(  # t-2番目までの観測でkernelをfitする
            X=normalized_params[..., :-2, :],
            Y=standarized_score_vals[:-2],
            is_categorical=(
                considered_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            initial_kernel_params=None,
            deterministic_objective=self._deterministic,
        )

        kernel_params_t1 = gp.fit_kernel_params(  # t-1番目までの観測でkernelをfitする
            X=normalized_params[..., :-1, :],
            Y=standarized_score_vals[:-1],
            is_categorical=(
                considered_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            initial_kernel_params=kernel_params_t2,
            deterministic_objective=self._deterministic,
        )

        kernel_params_t = gp.fit_kernel_params(  # t番目までの観測でkernelをfitする
            X=normalized_params,
            Y=standarized_score_vals,
            is_categorical=(
                considered_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            initial_kernel_params=kernel_params_t1,
            deterministic_objective=self._deterministic,
        )

        theta_t_star_index = np.argmax(standarized_score_vals)
        theta_t1_star_index = np.argmax(standarized_score_vals[:-1])
        theta_t_star = normalized_params[theta_t_star_index, :]
        theta_t1_star = normalized_params[theta_t1_star_index, :]
        covariance_theta_t_star_theta_t1_star = _compute_gp_posterior_cov_two_thetas(
            considered_search_space,
            normalized_params,
            standarized_score_vals,
            kernel_params_t,
            theta_t_star_index,
            theta_t1_star_index,
        )

        mu_t1_theta_t, sigma_t1_theta_t = _compute_gp_posterior(
            considered_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            normalized_params[-1, :],
            kernel_params_t1,
        )
        mu_t1_theta_t_star, sigma_t1_theta_t_star = _compute_gp_posterior(
            considered_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            theta_t_star,
            kernel_params_t1,
        )
        mu_t2_theta_t1_star, _ = _compute_gp_posterior(
            considered_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            theta_t_star,
            kernel_params_t1,
            # Use kernel_params_t1 instead of t2.
            # Use t1 under the assumption that t1 and t2 are approximately the same.
            # This is because kernel should same when comparing difference of mus.
        )
        mu_t_theta_t_star, sigma_t_theta_t_star = _compute_gp_posterior(
            considered_search_space,
            normalized_params,
            standarized_score_vals,
            theta_t_star,
            kernel_params_t,
        )
        mu_t1_theta_t1_star, _ = _compute_gp_posterior(
            considered_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            theta_t1_star,
            kernel_params_t1,
        )
        sigma_t1_theta_t_squared = sigma_t1_theta_t**2

        sigma_t1_theta_t_star_squared = sigma_t1_theta_t_star**2
        sigma_t_theta_t_star_squared = sigma_t_theta_t_star**2

        y_t = standarized_score_vals[-1]
        kappa_t1 = RegretBoundEvaluator._compute_standardized_regret_bound(
            kernel_params_t1,
            considered_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            self._delta,
            rng=self._rng.rng
        )

        theorem1_delta_mu_t_star = mu_t2_theta_t1_star - mu_t1_theta_t_star

        alg1_delta_r_tilde_t_term1 = theorem1_delta_mu_t_star

        theorem1_v = math.sqrt(
            max(
                sys.float_info.min,
                sigma_t_theta_t_star_squared
                - 2.0 * covariance_theta_t_star_theta_t1_star
                + sigma_t1_theta_t_star_squared,
            )
        )
        theorem1_g = (mu_t_theta_t_star - mu_t1_theta_t1_star) / theorem1_v

        alg1_delta_r_tilde_t_term2 = theorem1_v * scipy_stats.norm.pdf(theorem1_g)
        alg1_delta_r_tilde_t_term3 = theorem1_v * theorem1_g * scipy_stats.norm.cdf(theorem1_g)

        _lambda = prior.DEFAULT_MINIMUM_NOISE_VAR
        eq4_rhs_term1 = 0.5 * math.log(1.0 + _lambda * sigma_t1_theta_t_squared)
        eq4_rhs_term2 = 0.5 * sigma_t1_theta_t_squared / (sigma_t1_theta_t_squared + _lambda**-1)
        eq4_rhs_term3 = (
            0.5
            * sigma_t1_theta_t_squared
            * (y_t - mu_t1_theta_t) ** 2
            / (sigma_t1_theta_t_squared + _lambda**-1) ** 2
        )

        alg1_delta_r_tilde_t_term4 = kappa_t1 * math.sqrt(
            0.5 * (eq4_rhs_term1 + eq4_rhs_term2 + eq4_rhs_term3)
        )

        return (
            alg1_delta_r_tilde_t_term1
            + alg1_delta_r_tilde_t_term2
            + alg1_delta_r_tilde_t_term3
            + alg1_delta_r_tilde_t_term4
        )


def _compute_gp_posterior(
    considered_search_space: gp_search_space.SearchSpace,
    X: np.ndarray,
    Y: np.ndarray,
    x_params: np.ndarray,
    kernel_params: KernelParamsTensor,
) -> tuple[float, float]:  # mean, var

    acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
        kernel_params=kernel_params,
        search_space=considered_search_space,
        X=X,  # normalized_params[..., :-1, :],
        Y=Y,  # standarized_score_vals[:-1],
    )
    mean, var = gp.posterior(
        acqf_params.kernel_params,
        torch.from_numpy(acqf_params.X),
        torch.from_numpy(
            acqf_params.search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        ),
        torch.from_numpy(acqf_params.cov_Y_Y_inv),
        torch.from_numpy(acqf_params.cov_Y_Y_inv_Y),
        torch.from_numpy(x_params),  # best_params or normalized_params[..., -1, :]),
    )
    mean = mean.detach().numpy().flatten()
    var = var.detach().numpy().flatten()
    assert len(mean) == 1 and len(var) == 1
    return float(mean[0]), float(var[0])


def _compute_gp_posterior_cov_two_thetas(
    considered_search_space: gp_search_space.SearchSpace,
    normalized_params: np.ndarray,
    standarized_score_vals: np.ndarray,
    kernel_params: KernelParamsTensor,
    theta1_index: int,
    theta2_index: int,
) -> float:  # cov

    if theta1_index == theta2_index:
        return _compute_gp_posterior(
            considered_search_space,
            normalized_params,
            standarized_score_vals,
            normalized_params[theta1_index],
            kernel_params,
        )[1]

    assert normalized_params.shape[0] == standarized_score_vals.shape[0]

    acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
        kernel_params=kernel_params,
        search_space=considered_search_space,
        X=normalized_params,
        Y=standarized_score_vals,
    )

    _, var = _posterior_of_batched_theta(
        acqf_params.kernel_params,
        torch.from_numpy(acqf_params.X),
        torch.from_numpy(
            acqf_params.search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        ),
        torch.from_numpy(acqf_params.cov_Y_Y_inv),
        torch.from_numpy(acqf_params.cov_Y_Y_inv_Y),
        torch.from_numpy(normalized_params[[theta1_index, theta2_index]]),
    )
    assert var.shape == (2, 2)
    var = var.detach().numpy()[0, 1]
    return float(var)


def _posterior_of_batched_theta(
    kernel_params: KernelParamsTensor,
    X: torch.Tensor,  # [len(trials), len(params)]
    is_categorical: torch.Tensor,  # bool[len(params)]
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    theta: torch.Tensor,  # [batch, len(params)]
) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: [(batch,)], var: [(batch,batch)])

    assert len(X.shape) == 2
    len_trials, len_params = X.shape
    assert len(theta.shape) == 2
    len_batch = theta.shape[0]
    assert theta.shape == (len_batch, len_params)
    assert is_categorical.shape == (len_params,)
    assert cov_Y_Y_inv.shape == (len_trials, len_trials)
    assert cov_Y_Y_inv_Y.shape == (len_trials,)

    cov_ftheta_fX = gp.kernel(is_categorical, kernel_params, theta[..., None, :], X)[
        ..., 0, :
    ]  # [batch,len(trials)]
    assert cov_ftheta_fX.shape == (len_batch, len_trials)
    cov_ftheta_ftheta = gp.kernel(is_categorical, kernel_params, theta[..., None, :], theta)[
        ..., 0, :
    ]  # [batch,batch]
    assert cov_ftheta_ftheta.shape == (len_batch, len_batch)
    assert torch.allclose(cov_ftheta_ftheta.diag(), gp.kernel_at_zero_distance(kernel_params))
    assert torch.allclose(cov_ftheta_ftheta, cov_ftheta_ftheta.T)

    mean = cov_ftheta_fX @ cov_Y_Y_inv_Y  # [batch]
    assert mean.shape == (len_batch,)
    var = cov_ftheta_ftheta - cov_ftheta_fX @ cov_Y_Y_inv @ cov_ftheta_fX.T  # [(batch, batch)]
    assert mean.shape == (len_batch,)
    assert var.shape == (len_batch, len_batch)

    # We need to clamp the variance to avoid negative values due to numerical errors.
    return mean, torch.clamp(var, min=0.0)
