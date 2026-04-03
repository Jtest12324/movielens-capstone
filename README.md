# movielens-capstone
HarvardX PH125.9x Data Science Capstone - MovieLens Project

## Project Files

This repository contains all required files for the HarvardX PH125.9x Data Science Capstone - MovieLens Project:

1. **movielens.R** - Complete R script implementing the recommendation system with regularized collaborative filtering
2. **movielens-report.Rmd** - R Markdown report with all required sections (introduction, methods, results, conclusion)
3. **PDF Report** - To generate the PDF from the Rmd file, open `movielens-report.Rmd` in RStudio and click "Knit to PDF"

## Final RMSE

The final model achieves **RMSE = 0.8643** on the holdout test set, meeting the project objective of < 0.86490.

## How to Run

1. Install required packages: tidyverse, caret, data.table, lubridate
2. Run `movielens.R` to execute the full analysis
3. Knit `movielens-report.Rmd` to generate the PDF report

## Model Approach

The final model uses:
- Regularized movie effects (lambda = 5.25)
- Regularized user effects
- Genre effects
- Year released effects

**Author**: Joy Roy (Jtest12324)
**Course**: HarvardX PH125.9x Data Science Capstone
**Date**: April 3, 2026
