# Data dictionary: hotel_bookings.csv

Source: Kaggle, *Hotel booking demand* (Antonio, Almeida and Nunes), 119,390 bookings from a city hotel and a resort hotel in Portugal, arrivals Jul 2015 to Aug 2017. Judgements below come from `notebooks/01_explore_data.ipynb` (finding numbers F1-F14 are defined there).

| Attribute | Type | Noise / data quality | Distribution | Usefulness for predicting cancellation |
|---|---|---|---|---|
| `hotel` | Categorical (2) | None | City 66%, Resort 34% | **Useful.** City cancels 41.7%, Resort 27.8%. Consider interaction with other features or one model per hotel. |
| `is_canceled` | Binary 0/1 | None | 37.0% canceled | **TARGET.** Mildly imbalanced, not severe. |
| `lead_time` | Integer, 0 to 737 days | Long right tail; 481 bookings > 500 days (plausible but rare) | Right-skewed (skew 1.3), spike at 0-1 | **Very useful.** Strongest numeric signal (corr +0.29). Cancel rate climbs from 10% (0-7 days) to 68% (> 1 year). |
| `arrival_date_year` | Integer, 2015-2017 | None | 3 values; data runs Jul 2015 to Aug 2017 | **Limited as a feature** (only 3 values, cannot generalise to a new year). **Valuable as a time index** for a time-based train/test split. |
| `arrival_date_month` | Categorical (ordinal), 12 names | None | Peak Jul/Aug; low Dec-Jan | **Moderate.** Cancel rate ranges 30.5% (Jan) to 41.5% (Jun). Must be ordered or converted to a number. |
| `arrival_date_week_number` | Integer, 1-53 | None | Roughly seasonal | **Redundant** with month. Pick one. |
| `arrival_date_day_of_month` | Integer, 1-31 | None | About uniform | **Low.** Correlation with target is about 0. Day-of-week (derived from the full date) is more meaningful. |
| `stays_in_weekend_nights` | Integer, 0-19 | Outliers (up to 19) | Right-skewed, most are 0-2 | **Low alone**; combine with week nights into `total_nights`. |
| `stays_in_week_nights` | Integer, 0-50 | Outliers (up to 50) | Right-skewed, most are 1-3 | **Low alone**; combine into `total_nights`. Stays > 14 nights cancel 55% (439 rows). |
| `adults` | Integer, 0-55 | Impossible values: 0 adults and up to 55 adults | At least half are exactly 2 | **Moderate.** Needs cleaning (0 adults and >10 are likely errors or group blocks). |
| `children` | Float (should be integer), 0-10 | 4 missing values | 93% are 0 | **Low.** Impute 0 for the 4 missing. |
| `babies` | Integer, 0-10 | Outliers (10 babies) | > 99% are 0 | **Low.** Turn into `has_babies` flag or merge into `total_guests`. |
| `meal` | Categorical (5) | `Undefined` and `SC` both mean 'no meal package' (9.9% combined) | BB 77%, HB 12%, SC 9%, FB 0.7%, Undefined 1% | **Low-moderate.** Merge `Undefined` into `SC`. FB looks high (60%) but only 798 rows. |
| `country` | Categorical (ISO-3 code), 177 values | 488 missing (0.4%) | Very top-heavy: PRT 41%, then GBR, FRA, ESP, DEU | **Useful but high-cardinality.** PRT cancels 56.6%, GBR/FRA/DEU 17-20%. Group rare countries into 'Other'. |
| `market_segment` | Categorical (8) | 2 'Undefined' rows | Online TA 47%, Offline TA/TO 20%, Groups 17% | **Very useful.** Groups cancel 61%, Direct 15%, Corporate 19%. |
| `distribution_channel` | Categorical (5) | 5 'Undefined' rows | TA/TO 82% | **Moderate, overlaps** with market_segment. Check redundancy before keeping both. |
| `is_repeated_guest` | Binary 0/1 | None | 3.2% are 1 | **Weak-moderate** (corr -0.08). Repeat guests cancel less. |
| `previous_cancellations` | Integer, 0-26 | Heavy tail | 94.6% are 0 | **Useful but odd.** Exactly 1 previous cancellation = 94% cancel; 2+ drops back to about 30%. See anomaly section. |
| `previous_bookings_not_canceled` | Integer, 0-72 | Heavy tail | 97% are 0 | **Weak.** Mostly zero. Candidate for a `has_history` flag. |
| `reserved_room_type` | Categorical (10 letters, anonymised) | None | A = 72%, D = 16% | **Weak-moderate.** Letters are anonymised so no ordering is known. |
| `assigned_room_type` | Categorical (12 letters) | None | Same as reserved | **Timing risk.** Real assignment happens at/after check-in. Differs from reserved on 12.5% of rows, and those cancel only 5.4% (vs 41.6%). |
| `booking_changes` | Integer, 0-21 | Heavy tail | Zero for most bookings | **Useful but timing risk.** 0 changes = 41% cancel, 1+ changes = 14-20%. Changes accumulate after booking. |
| `deposit_type` | Categorical (3) | Suspicious pattern, see anomaly section | No Deposit 88%, Non Refund 12%, Refundable 0.1% | **Very useful but suspicious.** Non Refund bookings cancel 99.4%, the opposite of what you would expect. |
| `agent` | Float ID (should be categorical), 333 IDs | 13.7% missing (means 'no agent', not unknown) | Long tail of agent IDs | **Moderate.** Treat as categorical with an explicit 'no agent' level. Group small agents. |
| `company` | Float ID (should be categorical), 352 IDs | 94.3% missing (means 'no company') | Almost always missing | **Low as an ID.** Convert to a `has_company` flag (66% of those bookings are Corporate). |
| `days_in_waiting_list` | Integer, 0-391 | Heavy tail | 97% are 0 | **Low-moderate.** Waitlisted bookings cancel 64% vs 36%. Also only known after booking. |
| `customer_type` | Categorical (4) | None | Transient 75%, Transient-Party 21% | **Moderate.** Group customers cancel 10%, Transient 41%. |
| `adr` | Float (average daily rate) | 1 negative, 1,959 zeros (1.6%), one extreme value of 5,400 | Right-skewed, median 94.6 | **Weak linearly** (corr +0.05). Zeros are mostly complimentary/free stays. Clip or cap the 5,400 outlier. |
| `required_car_parking_spaces` | Integer, 0-8 | None | 94% are 0 | **Timing risk / leakage-like.** Every booking with parking is a non-cancel (0% cancel rate). |
| `total_of_special_requests` | Integer, 0-5 | None | 59% are 0 | **Useful but timing risk.** 0 requests = 48% cancel, 1+ requests = 5-22%. Requests may be added after booking. |
| `reservation_status` | Categorical (Check-Out, Canceled, No-Show) | None | 63% Check-Out, 36% Canceled, 1% No-Show | **LEAKAGE. Drop.** It is the outcome written as text. Canceled + No-Show = is_canceled exactly. |
| `reservation_status_date` | Date | None | Jul 2015 to Sep 2017 | **LEAKAGE. Drop.** Date the final status was set. For 95% of canceled bookings it is the cancel date. |
