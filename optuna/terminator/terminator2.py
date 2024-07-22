from __future__ import annotations

import abc
import math
import sys
from typing import cast
from typing import TYPE_CHECKING
import warnings

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.study import StudyDirection
from optuna.study.study import Study
from optuna.terminator.terminator import BaseTerminator
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import scipy.stats as scipy_stats
    import torch

    import optuna._gp.acqf as acqf
    import optuna._gp.gp as gp
    import optuna._gp.optim_sample as optim_sample
    import optuna._gp.prior as prior
    import optuna._gp.search_space as gp_search_space
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    optim_sample = _LazyImport("optuna._gp.optim_sample")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    acqf = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")
    scipy_stats = _LazyImport("scipy.stats")


@experimental_class("4.0.0")
class Terminator2(BaseTerminator):
    """Automatic stopping mechanism for Optuna studies.

    This class implements an automatic stopping mechanism for Optuna studies, aiming to prevent
    unnecessary computation. The study is terminated when the decrease in the
    'expected minimum simple regrets'—a quantitative measure of the potential for further
    optimization—between successive trials falls below a predefined threshold.

    For further information about the algorithm, please refer to the following paper:

    - `H. Ishibashi et al. A stopping criterion for Bayesian optimization by the gap of
    expected minimum simple regrets.
      <https://proceedings.mlr.press/v206/ishibashi23a.html>`__

    Args:
        min_n_trials:
            The minimum number of trials before termination is considered. Defaults to ``20``.
        threshold_ratio_to_initial_median:
            Set the threshold as ``threshold_ratio_to_initial_median`` times the median of the regrets
            of the first ``min_n_trials`` trials. Defaults to ``0.01``.
        deterministic_objective:
            Whether the objective function is deterministic or not.
            If :obj:`True`, the sampler will fix the noise variance of the surrogate model to
            the minimum value (slightly above 0 to ensure numerical stability).
            Defaults to :obj:`False`.

    Raises:
        ValueError: If ``min_n_trials`` is not a positive integer or less than ``3``.
        ValueError: If ``threshold_ratio_to_initial_median`` is not a positive number.

    Example:

        .. testcode::

            import logging
            import sys

            import optuna
            from optuna.terminator.terminator2 import Terminator2


            def rosenbrock(x):
                y = 0
                for d in range(len(x) - 1):
                    y += 100 * (x[d + 1] - x[d] ** 2) ** 2 + (x[d] - 1) ** 2
                return y


            study = optuna.create_study(direction="maximize")
            terminator = Terminator2()
            min_n_trials = 20

            while True:
                trial = study.ask()

                x = [trial.suggest_float(f"x{i}", -2.0, 2.0) for i in range(5)]

                value = rosenbrock(x)
                logging.info(f"Trial #{trial.number} finished with value {value}.")
                study.tell(trial, value)

                if trial.number > min_n_trials and terminator.should_terminate(study):
                    logging.info("Terminated by Optuna Terminator2!")
                    break


    .. seealso::
        Please refer to :class:`~optuna.terminator.TerminatorCallback` for how to use
        the terminator mechanism with the :func:`~optuna.study.Study.optimize` method.

    """

    def __init__(
        self,
        min_n_trials: int = 20,
        threshold_ratio_to_initial_median: float = 0.001,
        deterministic_objective: bool = False,
        delta: float = 0.1,
    ) -> None:
        if min_n_trials <= 1 or not np.isfinite(min_n_trials):
            raise ValueError("`min_n_trials` is expected to be an integer more than two.")
        if threshold_ratio_to_initial_median <= 0.0 or not np.isfinite(
            threshold_ratio_to_initial_median
        ):
            raise ValueError("`threshold_ratio_to_initial_median` is expected to be a positive.")

        self._min_n_trials = min_n_trials
        self._threshold_ratio_to_initial_median = threshold_ratio_to_initial_median
        self._deterministic = deterministic_objective
        self._median_threshold = None
        self._delta = delta

    def _raise_error_if_multi_objective(self, study: Study) -> None:
        if study._is_multi_objective():
            raise ValueError(
                "If the study is being used for multi-objective optimization,"
                f"{self.__class__.__name__} cannot be used."
            )

    @staticmethod
    def _compute_gp_covariance_matern52(squared_distance: float) -> float:
        sqrt5d = math.sqrt(5 * squared_distance)
        exp_part = math.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        return val

    def _compute_gp_posterior_cov_two_thetas(
        self,
        internal_search_space: gp_search_space.SearchSpace,
        normalized_params: np.ndarray,
        standarized_score_vals: np.ndarray,
        kernel_params: KernelParamsTensor,
        theta1_index: int,
        theta2_index: int,
    ) -> float:  # cov

        if theta1_index == theta2_index:
            return self._compute_gp_posterior(
                internal_search_space,
                normalized_params,
                standarized_score_vals,
                normalized_params[theta1_index],
                kernel_params,
            )[1]

        assert normalized_params.shape[0] == standarized_score_vals.shape[0]

        # kernel_params = gp.fit_kernel_params(
        #     X=normalized_params,
        #     Y=standarized_score_vals,
        #     is_categorical=(
        #         internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        #     ),
        #     log_prior=prior.default_log_prior,
        #     minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        #     initial_kernel_params=None,
        #     deterministic_objective=self._deterministic,
        # )
        acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
            kernel_params=kernel_params,
            search_space=internal_search_space,
            X=normalized_params,
            Y=standarized_score_vals,
        )

        _, var = _posterior(
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

    def _compute_gp_posterior(
        self,
        internal_search_space: gp_search_space.SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        x_params: np.ndarray,
        kernel_params: KernelParamsTensor,
    ) -> tuple[float, float]:  # mean, var

        # if star_flag:
        #     best_params: np.ndarray | None = normalized_params[..., -1, :]
        #     normalized_params = normalized_params[..., :-1, :]
        # else:
        #     best_params = None

        # kernel_params = gp.fit_kernel_params(
        #     X=normalized_params[..., :-1, :],
        #     Y=standarized_score_vals[:-1],
        #     is_categorical=(
        #         internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        #     ),
        #     log_prior=prior.default_log_prior,
        #     minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        #     initial_kernel_params=None,
        #     deterministic_objective=self._deterministic,
        # )
        acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
            kernel_params=kernel_params,
            search_space=internal_search_space,
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

    def compute_criterion(self, trials: list[FrozenTrial], study: Study) -> float:

        def _gpsampler_infer_relative_search_space() -> dict[str, BaseDistribution]:
            search_space = {}
            for name, distribution in (
                optuna.search_space.IntersectionSearchSpace()
                .calculate(study)
                .items()
                # IntersectionSearchSpaceは全trialで共通して出現する軸を得るためのもの。
            ):
                if distribution.single():
                    continue
                search_space[name] = distribution

            return search_space

        search_space = _gpsampler_infer_relative_search_space()
        if search_space == {}:
            return sys.float_info.max

        len_trials = len(trials)
        len_params = len(search_space)

        (
            internal_search_space,
            normalized_params,  # shape:[len(trials), len(params)]
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        assert normalized_params.shape == (len_trials, len_params)

        _sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        score_vals = np.array([_sign * cast(float, trial.value) for trial in trials])

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

        _lambda = prior.DEFAULT_MINIMUM_NOISE_VAR

        kernel_params_t2 = gp.fit_kernel_params(  # t-2番目までの観測でkernelをfitする
            X=normalized_params[..., :-2, :],
            Y=standarized_score_vals[:-2],
            is_categorical=(
                internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
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
                internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
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
                internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            initial_kernel_params=kernel_params_t1,
            deterministic_objective=self._deterministic,
        )

        theta_t_star_index = np.argmin(standarized_score_vals)
        theta_t1_star_index = np.argmin(standarized_score_vals[:-1])
        theta_t_star = normalized_params[theta_t_star_index, :]
        theta_t1_star = normalized_params[theta_t1_star_index, :]
        covariance_theta_t_star_theta_t1_star = self._compute_gp_posterior_cov_two_thetas(
            internal_search_space,
            normalized_params,
            standarized_score_vals,
            kernel_params_t,
            theta_t_star_index,
            theta_t1_star_index,
        )

        mu_t1_theta_t, sigma_t1_theta_t = self._compute_gp_posterior(
            internal_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            normalized_params[-1, :],
            kernel_params_t1,
        )
        # mu_t2_theta_t1, sigma_t2_theta_t1 = self._compute_gp_posterior(
        #     internal_search_space,
        #     normalized_params[:-2, :],
        #     standarized_score_vals[:-2],
        #     normalized_params[-2, :],
        #     kernel_params_t2,
        # )
        mu_t1_theta_t_star, sigma_t1_theta_t_star = self._compute_gp_posterior(
            internal_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            theta_t_star,
            kernel_params_t1,
        )
        mu_t2_theta_t1_star, _ = self._compute_gp_posterior(
            internal_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            theta_t_star,
            kernel_params_t1,  # あえてt2ではなくt1を使う！　著者実装の工夫で、t1とt2がほぼ変わらないという仮定の下で常にt1を用いる。
        )
        mu_t_theta_t_star, sigma_t_theta_t_star = self._compute_gp_posterior(
            internal_search_space,
            normalized_params,
            standarized_score_vals,
            theta_t_star,
            kernel_params_t,
        )
        mu_t1_theta_t1_star, _ = self._compute_gp_posterior(
            internal_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            theta_t1_star,
            kernel_params_t1,
        )
        sigma_t1_theta_t_squared = sigma_t1_theta_t**2

        sigma_t1_theta_t_star_squared = sigma_t1_theta_t_star**2
        sigma_t_theta_t_star_squared = sigma_t_theta_t_star**2

        y_t = standarized_score_vals[-1]
        kappa_t1 = _compute_kappa(
            kernel_params_t1,
            internal_search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            beta=2 * np.log(len_params * len_trials**2 * np.pi**2 / (6 * self._delta)),
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

    def should_terminate(self, study: Study) -> bool:
        """Judge whether the study should be terminated based on the values."""

        self._raise_error_if_multi_objective(study)

        trials = study._get_trials(states=[TrialState.COMPLETE])  # list[FrozenTrial]

        if len(trials) < self._min_n_trials:
            return False

        if self._median_threshold is None:
            initial_criteria = []
            for i in range(3, self._min_n_trials + 1):
                initial_criteria.append(self.compute_criterion(trials[:i], study))
            initial_criteria.sort()
            self._median_threshold = initial_criteria[len(initial_criteria) // 2]
            self._median_threshold *= self._threshold_ratio_to_initial_median

        current_criterion = self.compute_criterion(trials, study)
        print(f"{current_criterion=} , {self._median_threshold=}")

        return self._median_threshold >= current_criterion


def _posterior(
    kernel_params: KernelParamsTensor,
    X: torch.Tensor,  # [len(trials), len(params)]
    is_categorical: torch.Tensor,  # bool[len(params)]
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    theta: torch.Tensor,  # [batch, len(params)]
) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: [(batch,)], var: [(batch,batch)])
    # Xで構築したガウス過程でthetaの事後分布を求める。thetaは定義域上のちょうど2点で、Xに含まれていなくてもよい。

    assert len(X.shape) == 2
    len_trials, len_params = X.shape
    assert len(theta.shape) == 2
    len_batch = theta.shape[0]
    assert len_batch == 2
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
    assert torch.allclose(cov_ftheta_ftheta[0, 1], cov_ftheta_ftheta[1, 0])

    # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
    # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
    mean = cov_ftheta_fX @ cov_Y_Y_inv_Y  # [batch]
    assert mean.shape == (len_batch,)
    var = cov_ftheta_ftheta - cov_ftheta_fX @ cov_Y_Y_inv @ cov_ftheta_fX.T  # [(batch, batch)]
    assert var.shape == (len_batch, len_batch)
    var = torch.clamp(var, min=0.0)

    assert mean.shape == (2,) and var.shape == (2, 2)
    return mean, var

    # # We need to clamp the variance to avoid negative values due to numerical errors.
    # return (mean, torch.clamp(var, min=0.0))


def _compute_kappa(
    kernel_params: KernelParamsTensor,
    _gp_search_space: gp_search_space,
    normalized_top_n_params: np.ndarray,
    standarized_top_n_values: np.ndarray,
    beta: float,
    optimize_n_samples: int = 2048,
    rng: np.random.RandomState | None = None,
) -> float:
    """
    Compute min(ucb) over the all trials minus min(lcb) over the entire search space.
    """

    # 疑問: Theorem1の、 "Pick \delta \in (0,1)"って何？ "UCB_{\delta}"みたいな下付き文字に使われてるけど

    # calculate min_ucb
    ucb_acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.UCB,
        kernel_params=kernel_params,
        search_space=_gp_search_space,
        X=normalized_top_n_params,
        Y=standarized_top_n_values,
        beta=beta,
    )
    # UCB over the all trials.
    standardized_ucb_value = np.min(
        acqf.eval_acqf_no_grad(ucb_acqf_params, normalized_top_n_params)
    )

    # calculate min_lcb
    lcb_acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.LCB,
        kernel_params=kernel_params,
        search_space=_gp_search_space,
        X=normalized_top_n_params,
        Y=standarized_top_n_values,
        beta=beta,
    )

    xs = gp_search_space.sample_normalized_params(optimize_n_samples, lcb_acqf_params.search_space, rng=rng)

    # LCB over the search space.
    standardized_lcb_value = min(
        acqf.eval_acqf_no_grad(lcb_acqf_params, normalized_top_n_params).min(),
        acqf.eval_acqf_no_grad(lcb_acqf_params, xs).min()
    )

    return standardized_ucb_value - standardized_lcb_value  # standardized regret bound
