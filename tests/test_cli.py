"""Tests for the command-line interface.

The CLI had no test coverage at all, and accumulated two user-facing defects
because of it: it advertised a benchmark problem that could not fail and so
always reported a probability of zero, and it reported a hard-coded version
string that drifted from the package's own.

These tests exercise each subcommand and pin the reference probabilities to the
Monte Carlo values in ``test_benchmark_ground_truth.py``, so a stale number
fails here rather than being printed to a user as fact.
"""

from __future__ import annotations

import re

import pytest

from safe_ice import __version__
from safe_ice.cli import main, run_benchmark

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI colour codes.

    Python 3.14 colourises argparse help output, which splits `usage: safe-ice`
    across escape sequences. Whether it does so also depends on the terminal and
    on NO_COLOR/FORCE_COLOR, so the tests compare the uncoloured text rather
    than depending on the environment.
    """
    return _ANSI.sub("", text)


class TestVersion:
    def test_reports_the_package_version(self, capsys) -> None:
        """--version must track the package, not a hard-coded string."""
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0

        out = plain(capsys.readouterr().out)
        assert __version__ in out


class TestHelp:
    def test_no_command_prints_help(self, capsys) -> None:
        assert main([]) == 0
        out = plain(capsys.readouterr().out)
        assert "usage: safe-ice" in out
        assert "demo" in out

    def test_epilogue_has_no_placeholder_url(self, capsys) -> None:
        """The help text used to point at a `your-username` placeholder."""
        assert main([]) == 0
        out = plain(capsys.readouterr().out)
        assert "your-username" not in out
        assert "DiogoRibeiro7/adaptive-importance-sampling-ice" in out


class TestBenchmarkListing:
    def test_lists_the_available_problems(self, capsys) -> None:
        assert main(["benchmark", "--list"]) == 0
        out = plain(capsys.readouterr().out)
        for name in ("four-mode", "three-mode", "two-mode"):
            assert name in out

    def test_offers_the_oscillator(self, capsys) -> None:
        """Withdrawn while it could not fail; restored with the real model."""
        assert main(["benchmark", "--list"]) == 0
        assert "oscillator" in plain(capsys.readouterr().out)

    def test_unknown_problem_is_reported(self, capsys) -> None:
        assert main(["benchmark", "not-a-problem"]) == 0
        out = plain(capsys.readouterr().out)
        assert "Unknown problem" in out
        assert "four-mode" in out  # still shows what is available


class TestBenchmarkRun:
    @pytest.mark.parametrize("problem", ["four-mode", "three-mode", "two-mode"])
    def test_runs_and_reports_a_probability(self, problem: str, capsys) -> None:
        assert (
            main(["benchmark", problem, "--samples", "200", "--iterations", "2"]) == 0
        )
        out = plain(capsys.readouterr().out)
        assert "Estimated failure probability" in out
        assert "Relative error" in out

    def test_reference_values_match_monte_carlo_ground_truth(self, capsys) -> None:
        """The printed references must be the measured ones.

        four-mode was quoted as 1.22e-5 against a measured 5.8e-5, and
        three-mode as 2.3e-3 against 3.5e-3.
        """
        expected = {
            "four-mode": 5.815e-05,
            "three-mode": 3.475e-03,
            "two-mode": 2.690e-03,
            "oscillator": 1.798e-03,
        }
        for problem, reference in expected.items():
            run_benchmark(problem, n_samples=200, max_iterations=2)
            out = plain(capsys.readouterr().out)
            assert f"{reference:.2e}" in out, f"{problem}: reference not printed"


class TestDemo:
    @pytest.mark.slow
    def test_demo_runs(self, capsys) -> None:
        assert main(["demo"]) == 0
        out = plain(capsys.readouterr().out)
        assert "Safe-ICE Algorithm Demonstration" in out
        assert "Estimated failure probability" in out


class TestAnalyze:
    def test_analyze_is_a_placeholder(self, capsys) -> None:
        """Records that the subcommand is not implemented yet."""
        assert main(["analyze"]) == 0
        assert "coming soon" in plain(capsys.readouterr().out)
