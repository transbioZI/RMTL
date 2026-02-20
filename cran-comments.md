## Resubmission

This is a resubmission. The previous CRAN version is 0.9.9.

### Changes in this version (1.0.0)

* Bumped version to 1.0.0 (> existing CRAN version 0.9.9)
* Fixed non-canonical CRAN URLs in README.md
* Fixed invalid PubMed URLs in vignette bibliography (http -> https)
* Cleaned up DESCRIPTION: removed stale CRAN-added fields
* Updated `.Rbuildignore` to exclude non-standard top-level files

## Test environments

* Local: macOS Sequoia 15.7.3, R 4.4.2
* Platform: aarch64-apple-darwin20

## R CMD check results

There were no ERRORs or WARNINGs.

There were 2 NOTEs:

1. `unable to verify current time` -- transient network issue during check.
2. `Skipping checking math rendering: package 'V8' unavailable` -- local environment only.

Both are local environment issues and will not appear on CRAN's check infrastructure.

## Downstream dependencies

There are currently no downstream dependencies for this package.
