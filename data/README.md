# Data Directory

## Required Data Format

Place your fantasy football data file as `cleaned_fantasy_football_data.xlsx` in this directory.

### Required Columns

Your Excel file must contain the following columns:

| Column | Description | Type |
|--------|-------------|------|
| Player | Player name | String |
| Year | Season year | Integer |
| Age | Player age | Integer |
| G | Games played | Integer |
| Tgt | Targets | Integer |
| Rec | Receptions | Integer |
| RecYds | Receiving yards | Integer |
| RecTD | Receiving touchdowns | Integer |
| TD | Total touchdowns | Integer |
| FantPt | Fantasy points (standard) | Float |
| FantPtHalf | Fantasy points (half-PPR) | Float |
| FantPos | Fantasy position | String |

### Sample Data

A sample data file `sample_data.xlsx` is included for reference.

### Data Requirements

1. **Position Filter**: Only Wide Receivers (WR) are used in the MVP
2. **Time Range**: Last 11 years of data recommended
3. **Completeness**: Missing values will be filled with 0
4. **Format**: Excel (.xlsx) format only

### Data Sources

You can obtain fantasy football data from:
- Pro Football Reference
- FantasyPros
- ESPN Fantasy
- NFL.com
- Custom scrapers

## Example Data Structure

```
Player      | Year | Age | G  | Tgt | Rec | RecYds | RecTD | TD | FantPt | FantPtHalf | FantPos
------------|------|-----|----|----|-----|--------|-------|----|---------|-----------|---------
Justin J.   | 2023 | 24  | 16 | 157| 100 | 1280   | 10    | 10 | 210.0   | 235.0     | WR
Tyreek Hill | 2023 | 29  | 16 | 171| 119 | 1799   | 13    | 13 | 285.9   | 322.9     | WR
Amon-Ra St. | 2023 | 24  | 16 | 164| 119 | 1515   | 10    | 10 | 249.5   | 289.5     | WR
```

## Notes

- The model automatically filters for WR position only
- Feature engineering creates additional calculated fields
- Missing or infinite values are handled automatically
- Data from the most recent season (e.g., 2024) is used for predictions