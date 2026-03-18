#!/usr/bin/env python3
"""
Download state-of-the-art benchmark results from the HIVE-COTE 2.0 (HC2) study.

Source: https://www.timeseriesclassification.com/HC2.php
Paper:  Middlehurst, M., Large, J., Flynn, M. et al. HIVE-COTE 2.0: a new meta
        ensemble for time series classification. Mach Learn 110, 3211–3243 (2021).
        https://doi.org/10.1007/s10994-021-06057-9

Outputs:
  - benchmarks_hc2/ucr_sota_average_over_30.csv  (112 UCR, nine classifiers, averaged over 30 resamples)
  - benchmarks_hc2/uea_sota_average_over_30.csv    (26 UEA, averaged over 30 resamples)
"""
from pathlib import Path

import urllib.request

HERE = Path(__file__).parent
OUT_DIR = HERE / "benchmarks_hc2"

# CSV with univariate results for nine classifiers on 112 UCR problems, averaged over 30 resamples
URL_UCR = "https://www.timeseriesclassification.com/results/HC2/SOTA-AverageOver30.csv"
# CSV with multivariate results for 26 UEA problems, averaged over 30 resamples
URL_UEA = "https://www.timeseriesclassification.com/results/HC2/MTSC-SOTA-AverageOver30.csv"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  {dest.name}")


def main() -> None:
    print("HIVE-COTE 2.0 benchmarks (HC2)")
    print("  Paper: Middlehurst et al., Mach Learn 110, 3211–3243 (2021)")
    print("  https://doi.org/10.1007/s10994-021-06057-9")
    print("  https://www.timeseriesclassification.com/HC2.php\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading UCR (112 problems, 9 classifiers, average over 30 resamples)...")
    download(URL_UCR, OUT_DIR / "ucr_sota_average_over_30.csv")

    print("Downloading UEA (26 problems, average over 30 resamples)...")
    download(URL_UEA, OUT_DIR / "uea_sota_average_over_30.csv")

    (OUT_DIR / "README.md").write_text("""# HIVE-COTE 2.0 (HC2) benchmark results

- **ucr_sota_average_over_30.csv** — 112 UCR univariate problems, nine classifiers, accuracy averaged over 30 resamples.
- **uea_sota_average_over_30.csv** — 26 UEA multivariate problems, averaged over 30 resamples.

**Source:** [timeseriesclassification.com/HC2.php](https://www.timeseriesclassification.com/HC2.php)

**Citation:** Middlehurst, M., Large, J., Flynn, M. et al. *HIVE-COTE 2.0: a new meta ensemble for time series classification.* Mach Learn 110, 3211–3243 (2021). https://doi.org/10.1007/s10994-021-06057-9
""", encoding="utf-8")
    print(f"  README.md (citation)")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
