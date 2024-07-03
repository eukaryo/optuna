from __future__ import annotations

import abc
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
    import torch

    import optuna._gp.acqf as acqf
    import optuna._gp.gp as gp
    import optuna._gp.prior as prior
    import optuna._gp.search_space as gp_search_space
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    acqf = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")


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

            from sklearn.datasets import load_wine
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold

            import optuna
            from optuna.terminator import Terminator2


            study = optuna.create_study(direction="maximize")
            terminator = Terminator2()
            min_n_trials = 20

            while True:
                trial = study.ask()

                X, y = load_wine(return_X_y=True)

                clf = RandomForestClassifier(
                    max_depth=trial.suggest_int("max_depth", 2, 32),
                    min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
                    criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
                )

                scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))

                value = scores.mean()
                logging.info(f"Trial #{trial.number} finished with value {value}.")
                study.tell(trial, value)

                if trial.number > min_n_trials and terminator2.should_terminate(study):
                    logging.info("Terminated by Optuna Terminator2!")
                    break

    .. seealso::
        Please refer to :class:`~optuna.terminator.TerminatorCallback` for how to use
        the terminator mechanism with the :func:`~optuna.study.Study.optimize` method.

    """

    def __init__(
        self,
        min_n_trials: int = 20,
        threshold_ratio_to_initial_median: float = 0.01,
        deterministic_objective: bool = False,
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

    def _raise_error_if_multi_objective(self, study: Study) -> None:
        if study._is_multi_objective():
            raise ValueError(
                "If the study is being used for multi-objective optimization,"
                f"{self.__class__.__name__} cannot be used."
            )

    def _compute_gp_posterior(
        self,
        direction: StudyDirection,
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> tuple[float, float]:

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _sign = -1.0 if direction == StudyDirection.MINIMIZE else 1.0
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

        standarized_score_vals = (score_vals - score_vals.mean()) / max(1e-10, score_vals.std())

        kernel_params = gp.fit_kernel_params(
            X=normalized_params[..., :-1, :],
            Y=standarized_score_vals[:-1],
            is_categorical=(
                internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            initial_kernel_params=None,
            deterministic_objective=self._deterministic,
        )
        acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
            kernel_params=kernel_params,
            search_space=internal_search_space,
            X=normalized_params[..., :-1, :],
            Y=standarized_score_vals[:-1],
        )
        mean, var = gp.posterior(
            acqf_params.kernel_params,
            torch.from_numpy(acqf_params.X),
            torch.from_numpy(
                acqf_params.search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            torch.from_numpy(acqf_params.cov_Y_Y_inv),
            torch.from_numpy(acqf_params.cov_Y_Y_inv_Y),
            torch.from_numpy(normalized_params[-1, :]),
        )
        assert np.prod(mean.detach().numpy().shape) == 1
        assert np.prod(var.detach().numpy().shape) == 1
        return float(mean.detach().numpy().flatten()[0]), float(var.detach().numpy().flatten()[0])

    def compute_criterion(self, trials: list[FrozenTrial], study: Study) -> float:

        def _gpsampler_infer_relative_search_space() -> dict[str, BaseDistribution]:
            search_space = {}
            for name, distribution in (
                optuna.search_space.IntersectionSearchSpace().calculate(study).items()
            ):
                if distribution.single():
                    continue
                search_space[name] = distribution

            return search_space

        search_space = _gpsampler_infer_relative_search_space()

        mu_t1_theta_t, sigma_t1_theta_t = self._compute_gp_posterior(
            study.direction, trials, search_space
        )
        mu_t2_theta_t1, sigma_t2_theta_t1 = self._compute_gp_posterior(
            study.direction, trials[:-1], search_space
        )
        sigma_t1_theta_t_squared = sigma_t1_theta_t**2
        sigma_t2_theta_t1_squared = sigma_t2_theta_t1**2


        assert False  # TODO
        return 0.0

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
