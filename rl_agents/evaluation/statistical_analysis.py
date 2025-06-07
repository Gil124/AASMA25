"""
Statistical Analysis Module for RL Agent Evaluation
Provides statistical validation with paired t-tests and confidence intervals
"""
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Statistical analysis tools for RL agent performance evaluation"""
    
    @staticmethod
    def paired_t_test(sample1: List[float], sample2: List[float], 
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform paired t-test between two samples
        
        Args:
            sample1: First sample (e.g., RL agent win rates)
            sample2: Second sample (e.g., baseline win rates)
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary with test results
        """
        if len(sample1) != len(sample2):
            raise ValueError("Samples must have equal length for paired t-test")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(sample1, sample2)
        
        # Effect size (Cohen's d for paired samples)
        differences = np.array(sample1) - np.array(sample2)
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "degrees_freedom": len(sample1) - 1,
            "significant": p_value < alpha,
            "effect_size_cohens_d": cohens_d,
            "mean_difference": np.mean(differences),
            "std_difference": np.std(differences, ddof=1),
            "alpha": alpha
        }
    
    @staticmethod
    def one_sample_t_test(sample: List[float], population_mean: float,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        One-sample t-test against a known population mean
        
        Args:
            sample: Sample data
            population_mean: Known or hypothesized population mean
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_1samp(sample, population_mean)
        
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        cohens_d = (sample_mean - population_mean) / sample_std
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "degrees_freedom": len(sample) - 1,
            "significant": p_value < alpha,
            "sample_mean": sample_mean,
            "population_mean": population_mean,
            "effect_size_cohens_d": cohens_d,
            "alpha": alpha,
            "better_than_population": sample_mean > population_mean
        }
    
    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for sample mean
        
        Args:
            data: Sample data
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        return stats.t.interval(
            confidence, 
            len(data) - 1,
            loc=np.mean(data),
            scale=stats.sem(data)
        )
    
    @staticmethod
    def proportion_confidence_interval(successes: int, trials: int, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for proportion (e.g., win rate)
        
        Args:
            successes: Number of successes
            trials: Total number of trials
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)        """
        alpha = 1 - confidence
        # Use normal approximation for binomial proportion confidence interval
        p_hat = successes / trials
        z_score = stats.norm.ppf(1 - alpha/2)
        se = np.sqrt(p_hat * (1 - p_hat) / trials)
        return (
            max(0, p_hat - z_score * se), 
            min(1, p_hat + z_score * se)
        )
    
    @staticmethod
    def wilcoxon_signed_rank_test(sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """
        Non-parametric Wilcoxon signed-rank test for paired samples
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Dictionary with test results
        """
        if len(sample1) != len(sample2):
            raise ValueError("Samples must have equal length")
        
        statistic, p_value = stats.wilcoxon(sample1, sample2)
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "test_type": "wilcoxon_signed_rank"
        }
    
    @staticmethod
    def effect_size_analysis(sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """
        Calculate various effect size measures
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Dictionary with effect size measures
        """
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        
        # Cohen's d (pooled standard deviation)
        pooled_std = np.sqrt(((len(sample1) - 1) * std1**2 + (len(sample2) - 1) * std2**2) / 
                           (len(sample1) + len(sample2) - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Glass's delta
        glass_delta = (mean1 - mean2) / std2
        
        # Hedges' g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(sample1) + len(sample2)) - 9))
        hedges_g = cohens_d * correction_factor
        
        return {
            "cohens_d": cohens_d,
            "glass_delta": glass_delta,
            "hedges_g": hedges_g,
            "mean_difference": mean1 - mean2,
            "interpretation": StatisticalAnalyzer._interpret_effect_size(abs(cohens_d))
        }
    
    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, 
                      power: float = 0.8) -> int:
        """
        Calculate required sample size for given effect size and power
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Type I error rate
            power: Desired statistical power
            
        Returns:
            Required sample size per group
        """
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    @staticmethod
    def comprehensive_comparison(rl_results: List[float], baseline_results: List[float],
                               baseline_name: str = "Baseline") -> Dict[str, Any]:
        """
        Comprehensive statistical comparison between RL agent and baseline
        
        Args:
            rl_results: RL agent performance data
            baseline_results: Baseline performance data
            baseline_name: Name of baseline for reporting
            
        Returns:
            Complete statistical analysis
        """
        logger.info(f"Performing comprehensive statistical comparison vs {baseline_name}")
        
        # Descriptive statistics
        descriptive = {
            "rl_mean": np.mean(rl_results),
            "rl_std": np.std(rl_results, ddof=1),
            "rl_median": np.median(rl_results),
            "baseline_mean": np.mean(baseline_results),
            "baseline_std": np.std(baseline_results, ddof=1),
            "baseline_median": np.median(baseline_results),
            "rl_sample_size": len(rl_results),
            "baseline_sample_size": len(baseline_results)
        }
        
        # Confidence intervals
        rl_ci = StatisticalAnalyzer.confidence_interval(rl_results)
        baseline_ci = StatisticalAnalyzer.confidence_interval(baseline_results)
        
        # Statistical tests
        if len(rl_results) == len(baseline_results):
            # Paired t-test
            paired_test = StatisticalAnalyzer.paired_t_test(rl_results, baseline_results)
            wilcoxon_test = StatisticalAnalyzer.wilcoxon_signed_rank_test(rl_results, baseline_results)
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(rl_results, baseline_results)
            paired_test = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "test_type": "independent_t_test"
            }
            wilcoxon_test = None
        
        # Effect size
        effect_size = StatisticalAnalyzer.effect_size_analysis(rl_results, baseline_results)
        
        return {
            "descriptive_statistics": descriptive,
            "confidence_intervals": {
                "rl_95_ci": rl_ci,
                "baseline_95_ci": baseline_ci
            },
            "statistical_tests": {
                "parametric": paired_test,
                "non_parametric": wilcoxon_test
            },
            "effect_size_analysis": effect_size,
            "practical_significance": {
                "improvement": descriptive["rl_mean"] > descriptive["baseline_mean"],
                "improvement_magnitude": descriptive["rl_mean"] - descriptive["baseline_mean"],
                "improvement_percentage": ((descriptive["rl_mean"] - descriptive["baseline_mean"]) / 
                                         descriptive["baseline_mean"] * 100)
            }
        }

class ReportStatistics:
    """Statistics formatted for academic report"""
    
    @staticmethod
    def format_statistical_summary(analysis: Dict[str, Any]) -> str:
        """Format statistical analysis for report text"""
        desc = analysis["descriptive_statistics"]
        test = analysis["statistical_tests"]["parametric"]
        effect = analysis["effect_size_analysis"]
        
        improvement = desc["rl_mean"] - desc["baseline_mean"]
        improvement_pct = (improvement / desc["baseline_mean"]) * 100
        
        significance_text = "statistically significant" if test["significant"] else "not statistically significant"
        
        return (
            f"The RL agent achieved a mean performance of {desc['rl_mean']:.3f} "
            f"(SD = {desc['rl_std']:.3f}) compared to the baseline mean of "
            f"{desc['baseline_mean']:.3f} (SD = {desc['baseline_std']:.3f}), "
            f"representing a {improvement_pct:+.1f}% improvement. "
            f"This difference was {significance_text} "
            f"(t({test['degrees_freedom']}) = {test['t_statistic']:.3f}, "
            f"p = {test['p_value']:.3f}, Cohen's d = {effect['cohens_d']:.3f})."
        )
    
    @staticmethod
    def latex_statistical_table(comparisons: Dict[str, Dict[str, Any]]) -> str:
        """Generate LaTeX table with statistical comparisons"""
        lines = [
            "\\begin{table}[H]",
            "\\centering", 
            "\\caption{Statistical Analysis of RL Agent Performance}",
            "\\label{tab:statistical_analysis}",
            "\\begin{tabular}{lrrrrrr}",
            "\\hline",
            "Opponent & RL Mean & Baseline Mean & Diff & t-stat & p-value & Cohen's d \\\\",
            "\\hline"
        ]
        
        for opponent, analysis in comparisons.items():
            desc = analysis["descriptive_statistics"]
            test = analysis["statistical_tests"]["parametric"]
            effect = analysis["effect_size_analysis"]
            
            lines.append(
                f"{opponent} & {desc['rl_mean']:.3f} & {desc['baseline_mean']:.3f} & "
                f"{effect['mean_difference']:+.3f} & {test['t_statistic']:.3f} & "
                f"{test['p_value']:.3f} & {effect['cohens_d']:.3f} \\\\"
            )
        
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
