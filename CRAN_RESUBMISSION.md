# CRAN Resubmission Notes

## Date: February 1, 2026

## Changes Made to Address CRAN Team Feedback

### Issue
CRAN team reported a NOTE about problems with news in `NEWS.md` in the 'package subdirectories' check. The package was missing a properly formatted `NEWS.md` file that complies with CommonMark specification as required by `?news`.

### Resolution
1. **Created `NEWS.md`** with proper CommonMark format:
   - Primary headings (`#`) with version number and ISO 8601 date
   - Secondary headings (`##`) for categories  
   - Proper spacing and structure as per `?news` documentation

2. **Updated `DESCRIPTION` file**:
   - Version bumped from `0.9` to `0.9.1`
   - Date updated to `2026-02-01`

### Verification
- NEWS.md parses correctly: ✓
  ```r
  db <- tools:::.build_news_db_from_package_NEWS_md('NEWS.md')
  # Result: Classes 'news_db_from_md', 'news_db' and 'data.frame': 2 obs.
  ```
- Package builds successfully: ✓ (`RMTL_0.9.1.tar.gz`)
- NEWS.md included in package: ✓

### Files Changed
- `NEWS.md` (new file)
- `DESCRIPTION` (version and date updated)

### Test Environment
- R version 4.5.2 (2025-10-31)
- Platform: aarch64-apple-darwin20.0.0
- Running under: macOS Sequoia 15.7.3

## Submission Checklist
- [x] NEWS.md created with proper format
- [x] Version bumped to 0.9.1
- [x] Date updated
- [x] Package builds without errors
- [x] NEWS format validated
- [x] Ready for CRAN resubmission before 2026-02-21 deadline
