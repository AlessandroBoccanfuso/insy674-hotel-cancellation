# INSY 674 Group Project: Hotel Booking Cancellation

McGill Desautels MMA, INSY 674, Fall 2026 · Group #10
Dataset: Kaggle [*Hotel booking demand*](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (119,390 bookings, 2 hotels in Portugal, Jul 2015 to Aug 2017; António, de Almeida & Nunes, CC BY 4.0).

This README is the documentation required by the assignment rubric. It follows the five checklist sections (Géron, *Hands-On Machine Learning*, Appendix B), says who owns each section, and points to the code.

> **Start here.** The official deliverable is **[`notebooks/FinalDeliverable_Hotel_Cancellation_Forecasting.ipynb`](notebooks/FinalDeliverable_Hotel_Cancellation_Forecasting.ipynb)**. It contains the whole project in one notebook, Sections 0 to 18, with the recorded outputs. The other notebooks in `notebooks/` are the team's working files, kept for history. You do not need them to follow or rerun the project.

## Results at a glance

| Task | What it does | Final result (reserved test period, May–Aug 2017) |
|---|---|---|
| Task 2: booking cancellation classifier | Flags bookings likely to cancel (cancellations and no-shows) | Extra Trees + MLP ensemble, threshold 0.45: ROC-AUC **0.804**, precision **0.609**, recall **0.520** (caught 3,079 of 5,926 cancellations) |
| Task 1: room-nights released per night | Forecasts room-nights freed by later cancellations, 30, 14, 7 and 1 days ahead | Mean absolute error **43% lower** than a historical-rate baseline, averaged over the four horizons |
| Task 3: cancellation timing | Predict when a cancellation happens | Not implemented |

## Team, ownership and where to find each section

Team: Boccanfuso, Alessandro · Dupuy, Eglantine · Guo, Dario · Jiang, Logan · Laferrière-Mailfait, Antonin · Nadeem, Asad · Wang, Yao (Winston)

| Checklist section | Owner | Status | In the final notebook | Working notebook |
|---|---|---|---|---|
| 1. Frame the Problem and Look at the Big Picture | Eglantine Dupuy | done | Section 0 | `notebooks/00_framing_problem.ipynb` |
| 2. Explore the Data | Alessandro Boccanfuso | done | Sections 1–10 | `notebooks/01_explore_data.ipynb` |
| 3. Prepare the Data | Asad Nadeem | done | Section 11 | `notebooks/02_data_feature_engneering.ipynb`, `notebooks/03_feature_engineering_and_model_shortlist.ipynb` |
| 4. Measure and Compare Performance | Logan Jiang, Dario Guo | done | Sections 12–14 | `notebooks/04_Measure_and_Compare_Performance.ipynb` |
| 5. Fine-Tune the System (and Task 1 forecast) | Antonin Laferrière-Mailfait | done | Sections 15–18 | `notebooks/05_Fine_Tune_and_Forecast_Antonin.ipynb` |

Notes on the working notebooks:
- Notebooks 02, 03 and 04 each start with a full copy of the exploration notebook (01), which is why they are 1 to 3.6 MB. The final notebook contains each step only once.
- Notebook 03 also contains the quick model shortlist (Section 12), which used seed 42. The final notebook uses the project seed, 67 (see Section 4 below).
- Notebook 05 was run in Google Colab and repeats the preparation code so it can run on its own. In the final notebook it reuses Sections 11 and 13 instead.

## Repository layout

```
data/raw/hotel_bookings.csv       original Kaggle file, never edited
data/README.md                    data rules
notebooks/FinalDeliverable_Hotel_Cancellation_Forecasting.ipynb   the official, complete notebook (Sections 0–18)
notebooks/00_ ... 05_*.ipynb      working notebooks, one per checklist section (history)
reports/figures/                  charts saved by the exploration code, reused in the slides
docs/data_dictionary.md           every attribute: type, noise, distribution, usefulness
docs/attribute_profile_computed.csv   computed attribute profile
requirements.txt                  Python packages
```

When rerun, Sections 15–17 also create `reports/fine_tuning/` for search checkpoints and fitted models. These `.joblib` files are git-ignored.

### Team rules (to avoid merge pain)

- **One notebook per person.** Only edit your own notebook. If you spot a problem in someone else's, leave them a comment instead of changing it.
- **Keep `main` working.** `main` is the official version of the project. It should always run from top to bottom without errors.
- **Use a branch for your changes.** A branch is your own copy to work in, so you can't break the official version. To make one, the git command is `git checkout -b yourname/what-youre-doing`.
- **When you're done, open a pull request.** This is a request on GitHub that says "I'm finished, please check this and add it to `main`." A teammate reviews it and clicks Merge.
- **Use the same random seed** (`random_state=67`) so everyone gets the same results.

---

## 1. Frame the Problem and Look at the Big Picture

*Final notebook: Section 0.*

### Our objective

Hotel rooms are extremely perishable: a room that goes unsold on a given night has zero value forever. Cancellations and no-shows free up rooms the hotel can no longer sell through its usual channels, and the hotel loses that revenue.

Our target customers are hotels and other agencies selling stays. Our goal is to help them recover that revenue without risking double bookings or undercutting their own full-price sales. We use a dataset of 119,390 bookings across two hotels to build a model that:

- forecasts the rooms freed up per hotel per night,
- predicts whether a booking will be cancelled, and
- predicts when that cancellation will happen.

Our model could in turn be used to develop a "Too Good To Go" solution for hotels:

- **Revenue manager dashboard:** expected empty rooms per night with a range (e.g., "likely 8–12 rooms free on Nov 14").
- **Booking-level risk flags:** which reservations are likely to cancel, so the hotel can confirm with the guest, ask for a deposit, or plan around it. Prior research found that bookings contacted by the hotel cancelled less (António, de Almeida & Nunes, 2017), so outreach is itself a lever.
- **Release recommendation:** when to push a room to our channel, at roughly what discount, and through which channel.
- **Donation tier:** a possible CSR or tax angle for the hotel. When we can confidently predict that a room truly won't sell, it could go to a fallback tier with a social goal, such as shelters, NGOs, people in need and government partnerships.

### Current solutions and workarounds

- **Revenue management systems (RMS):** already forecast demand and set prices.
- **Hotel policies:** to mitigate cancellations, hotels implement rigid cancellation policies and overbooking tactics, which can hurt revenue and reputation.
- **Last-minute marketplaces:** HotelTonight presents a selection of unsold rooms at steep discounts every day at noon local time. Several other last-minute booking products have entered the market. These sources date from 2011–2014, so the current market should be re-checked.
- **Academic ML work (same dataset authors):** a prototype deployed in two hotels trains a classifier daily to flag "likely to cancel" bookings and compute net demand. Related work by the same authors reports cancellation prediction accuracy above 90%.

**Our angle:** hotels rarely discount unsold rooms because they fear cannibalizing loyal guests who paid the full price. A forecast tied to a discreet resale channel may address that. In our limited search we did not find a product that combines cancellation forecasting with a resale channel; a fuller market scan would be needed to confirm this gap.

### How will we frame the problem?

**Supervised learning**, with three tasks:

1. **Task 1 (main deliverable):** forecast rooms freed per hotel per night, built by aggregating bookings.
2. **Task 2 (feeds Task 1):** binary classification on `is_canceled`.
3. **Task 3 (potential addition):** predict when a cancellation happens (days before arrival).

Design choices:

- **Offline / batch learning** to start (static historical data). In production, retrain on a schedule (e.g., daily).
- **Several lead times** (30, 14, 7 and 1 days out), since hotels decide at different points.
- **Leakage:** drop `reservation_status` and `reservation_status_date` as features, because they reveal the outcome.
- **Split by time, not randomly:** train on earlier arrivals and test on later ones. A random split leaks seasonality and inflates scores.

### How will performance be measured?

- **Forecast:** MAE on rooms freed per night, reported by lead time (accuracy at 7–14 days matters more than at 200 days).
- **Prediction interval coverage:** does "8–12 rooms" contain the truth about as often as claimed?
- **Classifier:** ROC-AUC, plus precision and recall, with emphasis on precision (falsely predicting a free room leads to double bookings). Not plain accuracy: predicting "never cancels" already scores about 63% on the raw data and about 71% on the cleaned validation period.
- **Calibration** (Brier score, reliability plot), since pricing decisions use the probability itself.
- **Business metric:** revenue recovered per month vs. the hotel's current approach, net of discounts given and any walked guests.

### Is the performance measure aligned with the objective?

Largely, since the hotel acts on per-night counts, which is what we measure. Some gaps remain:

- **Asymmetric costs:** overestimating free rooms (double booking) likely costs more than underestimating (missed resale). We considered an asymmetric loss or a cost-based threshold.
- **Timing:** a correct prediction 3 weeks out is worth more than one the day before. Evaluating by lead time partly addresses this.
- **Model accuracy ≠ business value:** high scores don't prove hotels will act on recommendations. That needs a pilot or A/B test.

### What we delivered compared with this plan

| Planned element | What was delivered | Final notebook |
|---|---|---|
| Task 1: forecast rooms freed per hotel per night | Forecast of **room-nights released by later explicit cancellations** among reservations still open, per hotel-night, 30, 14, 7 and 1 days ahead. Built by summing booking-level cancellation probabilities from a snapshot-aware logistic regression, not a time-series model. No-shows are excluded because they cannot be resold in advance. | Sections 16, 17.1 |
| Task 2 feeds Task 1 | Task 1 uses its own conditional model, reusing the tuned logistic-regression configuration from Task 2 plus two timing features (days until arrival, days since booking). | Section 16.1 |
| Task 2: classify `is_canceled` | Extra Trees + MLP ensemble, decision threshold 0.45. The target includes no-shows. | Sections 12–15, 17.1 |
| Task 3: cancellation timing | **Not implemented.** | – |
| Prediction interval coverage | Nominal 90% empirical bands from validation residuals. | Sections 16.2, 17.1 |
| Calibration | Brier score and reliability plot on the validation period; calibration is imperfect. | Section 15.4 |
| Asymmetric costs | Approximated by choosing the threshold that maximises F0.5 (precision-weighted), not by a cost function. | Section 15.4 |
| Business metric: revenue recovered | **Not measured.** The data has no capacity, resale prices or cost information. | Section 18.6 |
| Release recommendation, donation tier | Product ideas only; not modelled. | – |

## 2. Explore the Data

*Final notebook: Sections 1–10. Column-by-column details: [`docs/data_dictionary.md`](docs/data_dictionary.md).*

Each finding below carries a label (F3, F6, …) that matches a takeaway in the notebook.

1. **Leakage columns.** `reservation_status` and `reservation_status_date` are dropped as features because they contain what we are trying to predict: `reservation_status` spells out Canceled or Check-Out (F6). `reservation_status_date` is still used to build historical snapshots for Task 1, but never as an input feature.
2. **Duplicates.** 26.8% of rows are exact duplicates. We remove them before splitting into training and test sets, because identical rows could fall into both and show the model the answer. Duplicates also cancel more (63% vs 27.5%), which would inflate the cancel rate (F3). Caveat: if these rows are really separate bookings (e.g., several identical rooms booked together), dropping them undercounts demand. There is no booking ID, so we cannot tell; Task 1 reports a sensitivity check (see decision 2).
3. **Lead time is the cleanest driver:** 10% cancel when booked within a week, 68% when booked over a year ahead (F9).
4. **Who books matters:** Groups (61%) and Portuguese guests (57%) cancel most; Direct (15%) and Corporate (19%) least (F9).
5. **`deposit_type = Non Refund` cancels 99%,** which goes against common sense: people who lose money by cancelling should cancel less. Almost all are Portuguese group or agency blocks and duplicate rows, so it is likely a record-keeping artifact (F11).
6. **Columns known only after booking** (parking, assigned room, booking changes, special requests). The file only stores each booking's final values, so we cannot know what these looked like 30 days before arrival. They are excluded from all models (see the timing table in Section 5 of the final notebook).
7. **Otherwise clean:** 4 columns have missing values (company, agent, country, children); for `agent` and `company`, missing means "none", which is real information (F2).

**Team decisions** (status after the final evaluation)

| # | Decision | Status |
|---|---|---|
| 1 | Prediction moment | **Settled:** the Task 1 forecast is made 30, 14, 7 and 1 days before each night. Task 2 is a booking-level benchmark on final stored values. |
| 2 | Drop exact duplicates? | **Settled:** dropped for the classifier. Task 1 drops them in its main analysis and reports a sensitivity check with duplicates counted separately (final notebook, Section 16.2). |
| 3 | Keep `deposit_type`? | Kept. The planned with/without comparison was **not run**; it is listed as a limitation. |
| 4 | Random vs time-based split | **Settled:** split by time (train on earlier arrivals, test on later). Time-ordered cross-validation inside the training period. |
| 5 | Metric | **Settled:** MAE by lead time and interval coverage for the forecast; ROC-AUC, precision and calibration for the classifier. Baseline: "never cancels" scores 63% on raw data (72.5% once duplicates are removed). Revenue recovered could not be measured with this data. |

## 3. Prepare the Data

*Final notebook: Section 11.*

We implemented the transformation ideas from exploration (Section 9). From 119,390 original records, we removed 31,994 exact duplicate rows, 166 records with no guests, and one record with a negative average daily rate (ADR). This left **87,229 bookings**, with a cancellation rate of 27.5%. Zero-ADR and zero-night bookings were kept, with a flag identifying zero-night stays. The target is `is_canceled` (1 = cancelled, 0 = not cancelled).

We sorted the data by arrival date and gave the earliest 70% of unique arrival dates to training, the next 15% to validation, and the last 15% to testing. These percentages refer to dates, not booking counts.

| Split | Bookings | Arrival dates | Cancellation rate |
|---|---:|---|---:|
| Train | 56,132 | 2015-07-01 to 2017-01-05 | 25.1% |
| Validation | 14,009 | 2017-01-06 to 2017-05-04 | 28.6% |
| Test | 17,088 | 2017-05-05 to 2017-08-31 | 34.7% |

The cancellation rate rises over time, so the test period is harder than the training period.

After cleaning and splitting, features were built with `BookingFeatureEngineer`, a `ColumnTransformer` and a scikit-learn `Pipeline`:

- **Feature selection:** we excluded `reservation_status` and `reservation_status_date`, which reveal the outcome, plus `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces` and `total_of_special_requests`, whose timing creates a leakage risk. Raw `company` IDs were replaced by `has_company`; `distribution_channel` was excluded because it overlaps with `market_segment`.
- **Missing and categorical values:** missing `children` → 0, missing `country` → `UNKNOWN`, missing `agent` → `NO_AGENT`. We created `has_agent` and `has_company`, merged undefined meal categories into `SC`, and mapped undefined market segments to `UNKNOWN`.
- **Feature engineering:** total nights, total guests, family and baby indicators, weekend-stay ratio, ADR per guest, booking-history indicators and historical cancellation rate; lead-time buckets; log transforms of lead time, ADR and total nights; arrival weekday and weekend indicators; cyclical month features; and hotel × market-segment interactions. Historical variables are assumed to describe bookings made before the prediction time.
- **Learned preprocessing:** selected numeric features were capped at their training-set 99.5th percentiles, median-imputed and standardised. Categorical features were one-hot encoded, grouping categories that appear fewer than 100 times in the fitting data.

All learned cutoffs, medians, scaling parameters and category groupings were fitted on training data only. During cross-validation, the complete preprocessing-and-model pipeline was cloned and refitted within each training fold. The pipeline fitted on the full training period produced 171 encoded features. `deposit_type` was kept in all reported experiments.

## 4. Measure and Compare Performance

*Final notebook: Sections 12–14.*

**Quick shortlist (Section 12).** Nine models (including a majority-class baseline) were trained with default settings on the same 15,000-booking training subset and scored on the validation period. This gives a first, cheap ranking. It uses the project seed (67); the working notebook 03 used seed 42. Changing the seed moves some scores by about 0.01 and changes the order (with seed 42 the MLP tied for first; with seed 67 Random Forest enters the top three). This instability is why the shortlist was decided by cross-validation instead.

**Cross-validated comparison (Sections 13–14).** We compared eight algorithms covering linear, margin-based, tree-ensemble, probabilistic and neural-network approaches. All experiments used the same five chronological, expanding-window folds within the training period, with `random_state=67` where applicable. Each fold trained on earlier arrival dates and evaluated on the following period, keeping bookings with the same arrival date together. We computed the mean and standard deviation of ROC-AUC, precision, recall, F1 and Brier score across folds. ROC-AUC was the main ranking metric; Brier score is unavailable for Linear SVM because its decision scores are not probabilities. The test set was reserved for the final evaluation.

For each algorithm, we evaluated three feature versions with unchanged model settings: V0 used the baseline features; V1 removed `is_domestic`, which overlaps with `country`; and V2 added `estimated_stay_value = adr × total_nights` to V1. This produced 24 model–feature combinations. Improvements were not consistent across algorithms: Extra Trees improved from 0.8111 to 0.8170 mean AUC between V0 and V2, while MLP kept its highest mean AUC with V0.

The highest-mean-AUC version of each algorithm is shown below. False-positive and false-negative rates come from the separate validation set, using a probability threshold of 0.5 (the standard decision boundary for Linear SVM).

| Rank | Model | Version | CV AUC ± SD | CV precision | CV recall | CV F1 | CV Brier | Val FP rate | Val FN rate |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Extra Trees | V2 | 0.8170 ± 0.0778 | 0.7634 | 0.2642 | 0.3625 | 0.1383 | 2.51% | 78.79% |
| 2 | Neural Net (MLP) | V0 | 0.8130 ± 0.0741 | 0.7096 | 0.2940 | 0.4002 | 0.1493 | 10.81% | 55.78% |
| 3 | Random Forest | V1 | 0.8111 ± 0.0782 | 0.7750 | 0.2448 | 0.3428 | 0.1410 | 2.48% | 80.41% |
| 4 | Linear SVM | V0 | 0.8021 ± 0.0793 | 0.6615 | 0.2501 | 0.3422 | n/a | 4.16% | 75.39% |
| 5 | Logistic Regression | V1 | 0.7984 ± 0.0871 | 0.6636 | 0.2734 | 0.3693 | 0.1482 | 5.44% | 71.35% |
| 6 | SGD Logistic | V2 | 0.7959 ± 0.0860 | 0.6703 | 0.2665 | 0.3461 | 0.1556 | 5.46% | 75.89% |
| 7 | Histogram Gradient Boosting | V2 | 0.7928 ± 0.0919 | 0.7070 | 0.3040 | 0.3946 | 0.1525 | 9.27% | 56.95% |
| 8 | Gaussian Naive Bayes | V2 | 0.7213 ± 0.1056 | 0.3676 | 0.7190 | 0.4848 | 0.3469 | 35.53% | 29.00% |

We assessed important variables using validation-set permutation importance: the drop in ROC-AUC after shuffling each engineered feature, repeated five times. **Agent, lead time and country** were the most important features across models. Removing `is_domestic` often increased the measured importance of `country`, consistent with their overlapping information. The added stay-value feature did not give a consistent improvement.

Confusion matrices, subgroup comparisons by hotel and market segment, and confidently wrong predictions showed different error patterns. Extra Trees and Random Forest raised few false alarms but missed many cancellations. MLP and Histogram Gradient Boosting reduced missed cancellations at the cost of more false alarms. Gaussian Naive Bayes caught more cancellations but wrongly flagged 35.53% of non-cancelled validation bookings. Some missed cancellations were last-minute bookings with no previous cancellations, which look safe. Some false alarms were repeat guests with a cancellation history. Resort Hotel showed higher missed-cancellation rates in several comparisons. Guest identifiers and prior behaviour, detailed cancellation terms, payment information and relative market prices, where available at prediction time, could help separate these cases.

We shortlisted **Extra Trees V2, MLP V0, Logistic Regression V1 and Histogram Gradient Boosting V2**. Extra Trees had the highest mean CV AUC; MLP combined a strong AUC with a different error pattern; Logistic Regression provided an interpretable linear probability model; and Histogram Gradient Boosting offered a different false-positive / false-negative trade-off. Validation error-set overlap (Jaccard) was 56.2% between Extra Trees and MLP, and 61.0% between Extra Trees and Histogram Gradient Boosting. By comparison, Random Forest overlapped 89.2% with Extra Trees, and Linear SVM overlapped 91.6% with Logistic Regression. SGD Logistic offered no clear advantage, and Gaussian Naive Bayes had a weaker AUC and many more false alarms. Selection therefore weighed performance, interpretability and error diversity together. Variation across time folds was substantial, so small differences in mean scores were interpreted cautiously.

## 5. Fine-Tune the System

*Final notebook: Sections 15–18. Owner: Antonin Laferrière-Mailfait.*

### 5.1 Scope and evaluation design

This section continues from the preparation, chronological model comparison and four-model shortlist above. Task 2 predicts the binary `is_canceled` outcome, which includes both explicit cancellations and no-shows. The Task 1 implementation forecasts room-nights released by later explicit pre-arrival cancellations among observed reservations.

For Task 2, we kept the cleaning rules, feature versions and chronological partitions from Section 3: 56,132 training bookings, 14,009 validation bookings and 17,088 test bookings. Five expanding-window folds were built within the training period using unique arrival dates. Feature engineering, imputation, outlier capping, category encoding and scaling were fitted within each fold's pipeline.

Mean CV ROC-AUC was the model-selection criterion. We also recorded fold standard deviation, precision, recall, F1 and Brier score. Classification metrics first used the default 0.5 decision boundary. CV means are averages of the five fold metrics; the standard deviations describe variation across periods, not confidence intervals.

### 5.2 Randomized tuning of models and transformations

We used randomized parameter sampling with the same five chronological folds and seed 67, saving results after every completed fold. An initial broader Extra Trees search was interrupted because individual fits were too expensive; its incomplete results were not used. The completed search used a restricted computing budget.

| Shortlisted configuration | Randomized alternatives | Untuned reference also evaluated? | Original mean CV AUC | Retained mean CV AUC |
|---|---:|---|---:|---:|
| Extra Trees V2 | 6 | Yes | 0.8170 | 0.8170 |
| Logistic Regression V1 | 3 | Yes | 0.7984 | 0.7993 |
| Histogram Gradient Boosting V2 | 2 | Yes | 0.7932 | 0.7983 |
| MLP V0 | 2 | Yes | 0.8130 | 0.8130 |

Histogram Gradient Boosting's original score is 0.7932 here and 0.7928 in Section 4. The tuning was run in Google Colab with a different scikit-learn version (see *Running it*); the small difference comes from that.

The Extra Trees search varied depth, minimum leaf size and class weighting, while fixing 90 trees and square-root feature sampling to control runtime. It also explored no outlier capping versus 99th- and 99.5th-percentile capping, median versus mean numeric imputation, and rare-category minimum frequencies. These imputation choices apply to missing values that reach the numeric imputers; they do not override the domain-based replacements already made in the feature engineer.

Logistic regression tuning explored regularization strength, class weighting and rare-category grouping. Boosting tuning explored learning rate, iteration count, leaf count, minimum leaf size and L2 regularization. MLP tuning explored hidden-layer size, weight regularization and initial learning rate, keeping the existing 90-iteration maximum and early stopping.

The retained configurations were:

- **Extra Trees V2:** original settings: 90 trees, maximum depth 18, minimum leaf size 4, square-root feature sampling, no class weighting. None of the six alternatives beat its original mean CV AUC.
- **Logistic Regression V1:** `C = 0.4340645213116581`, no class weighting, rare-category minimum frequency 100. The AUC improvement was small.
- **Histogram Gradient Boosting V2:** learning rate 0.05, 60 maximum iterations, 15 maximum leaf nodes, minimum leaf size 50, L2 regularization 0.
- **MLP V0:** one hidden layer of 48 neurons, `alpha = 0.01142030649919016`, initial learning rate 0.001. Its AUC improvement was negligible at four decimals.

These were bounded searches, not evidence of exhaustive optimization. We kept the original settings whenever the alternatives did not improve the declared criterion.

### 5.3 Ensemble comparison

We evaluated two predefined equal-weight soft-voting ensembles, averaging cancellation probabilities for the same held-out bookings in each chronological fold. Member pipelines were fitted independently within each fold.

| System | Mean CV ROC-AUC | Fold SD | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Extra Trees + MLP** | **0.8214** | 0.0737 | 0.7440 | 0.2743 | 0.3796 | 0.1408 |
| All four models | 0.8203 | 0.0743 | 0.7431 | 0.2640 | 0.3662 | 0.1420 |
| Extra Trees V2 | 0.8170 | 0.0778 | 0.7634 | 0.2642 | 0.3625 | **0.1383** |
| MLP V0 | 0.8130 | 0.0736 | 0.7068 | 0.2919 | 0.3971 | 0.1492 |
| Logistic Regression V1 | 0.7993 | 0.0863 | 0.6695 | 0.2693 | 0.3671 | 0.1478 |
| Histogram Gradient Boosting V2 | 0.7983 | 0.0836 | 0.7371 | 0.2670 | 0.3590 | 0.1490 |

We selected the Extra Trees + MLP ensemble by mean CV ROC-AUC. Its 0.0044 gain over Extra Trees means slightly better risk ranking, while Extra Trees kept the lowest Brier score. The ensemble did not win on every measure.

### 5.4 Validation, threshold choice and final Task 2 evaluation

The selected ensemble was fitted on the training partition and evaluated on the separate validation period: ROC-AUC 0.8346, Brier score 0.1474. A reliability plot showed imperfect calibration, including underprediction in several middle-to-high probability groups.

Because the project emphasises avoiding false cancellation flags, we chose the decision threshold that maximises validation F0.5 over the fixed grid 0.05, 0.10, …, 0.95. F0.5 weights precision more than recall, but it is a development proxy, not an estimated hotel cost function. The selected threshold was **0.45** (rule: predicted probability > 0.45).

After freezing the configurations, equal ensemble weights and threshold, we refitted both member pipelines on all 70,141 development bookings (train + validation) and evaluated the reserved test partition once.

| System / decision rule | Test ROC-AUC | Brier | Precision | Recall | F1 | F0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Development-rate probability baseline | 0.5000 | 0.2344 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Original Extra Trees configuration, threshold 0.50 | 0.7961 | 0.1734 | 0.6763 | 0.2933 | 0.4091 | 0.5362 |
| Selected ensemble, threshold 0.50 | 0.8040 | 0.1703 | 0.6391 | 0.3966 | 0.4894 | 0.5694 |
| **Final ensemble, threshold 0.45** | **0.8040** | **0.1703** | **0.6090** | **0.5196** | **0.5607** | **0.5887** |

The final operating point produced 9,185 true negatives, 1,977 false positives, 2,847 false negatives and 3,079 true positives. Compared with the ensemble at threshold 0.50, it caught 729 more cancellations/no-shows with 650 more false positives. Changing the threshold affects decisions, not ROC-AUC or Brier score.

The final ensemble improved test AUC and probability error compared with the original Extra Trees configuration. However, precision of 60.9% means about 39.1% of cancellation flags were false alarms. **The system supports prioritising which guests to contact or confirm; the results do not justify automatically releasing flagged rooms for resale.** No model or threshold changes were made after seeing these test results.

### 5.5 Task 1: room-nights released by cancellations

- **Forecast target.** Later explicit pre-arrival cancellations among reservations still open at a historical prediction date and covering a particular hotel-night. This is not total hotel availability: capacity, replacement bookings and full inventory were unavailable. No-shows are excluded.
- **Snapshot construction.** Booking date was estimated as arrival date minus lead time. Stays were expanded into nights, excluding departure day. A booking-night was eligible if it had already been booked, arrival was still in the future, and its final status had not yet been recorded at the end of the prediction date. Dates were treated as end-of-day cutoffs because exact timestamps were unavailable.
- **Time separation.** Training nights ran from July 31, 2015 to January 5, 2017, using only outcomes recorded by that fitting cutoff. The validation-night window was February 5 – May 4, 2017, so that even 30-day forecast origins came after the training cutoff. Training contained 382,025 booking-night-horizon rows, reduced to 368,660 distinct booking/origin observations for fitting.
- **Conditional model.** We reused the retained logistic-regression configuration and fitted fresh weights to the snapshot target, adding days until arrival and days since booking. Outcome/status fields and the timing-risk columns were withheld. This was one fixed fit, not another hyperparameter search.
- **Forecast and baselines.** Probabilities were summed by hotel, night and horizon, and compared with training-period cancellation rates by hotel/horizon and a zero-cancellation forecast. The 0.45 threshold was not applied to the count forecast.
- **Validation policy.** We selected the lowest-MAE method per horizon, with RMSE as the tie-breaker. The zero forecast was selected at one day; the snapshot model at seven, fourteen and thirty days. The zero forecast is a conservative choice that minimises MAE, not evidence that cancellations are impossible.
- **Duplicate sensitivity.** Holding the model, rates and methods fixed while counting identical records separately, validation MAE rose from 0.3090 to 0.4045 (1 day), 2.1291 to 2.6552 (7 days), 2.4066 to 3.0429 (14 days) and 3.6254 to 4.4869 (30 days). This shows the counting assumption matters; it does not tell us whether duplicate rows are database repeats or genuine separate reservations.
- **Final refit and evaluation.** The model was refitted on 451,745 distinct booking/origin observations with outcomes recorded by May 4, 2017. Test nights ran from June 4 to August 31, 2017; even the earliest 30-day test origin came after the final fitting cutoff. Evaluation covered 178 hotel-nights per horizon (712 hotel-night-horizon records).

| Horizon | Frozen method | Test MAE | Historical-rate MAE | Zero-forecast MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|---:|
| 1 day | Zero forecast | 0.2809 | 0.4652 | 0.2809 | 0.5996 | −0.2809 |
| 7 days | Snapshot model | 1.7456 | 3.2066 | 2.4045 | 2.1376 | +1.1221 |
| 14 days | Snapshot model | 2.1857 | 4.5971 | 6.6236 | 2.8446 | −0.9854 |
| 30 days | Snapshot model | 3.9414 | 6.0855 | 13.2921 | 5.0549 | +1.0784 |

Bias is forecast minus actual. Averaging equally across the four horizons, the frozen policy's test MAE was about **2.0384**, compared with **3.5886** for historical rates: about **43.2% lower error**. These averages are calculated from the rounded results shown.

**Uncertainty bands.** We used hotel/horizon-specific 5th and 95th percentiles of validation residuals (actual minus forecast), rounded count bounds outward, and clipped lower bounds at zero. These are nominal 90% empirical bands, not formal coverage guarantees.

| Horizon | Test coverage | Mean band width (room-nights) |
|---|---:|---:|
| 1 day | 98.31% | 1.5000 |
| 7 days | 99.44% | 5.9607 |
| 14 days | 94.38% | 9.2360 |
| 30 days | 93.82% | 15.6404 |

Coverage exceeded 90% in the test period, with fairly wide bands, especially at thirty days. The bands were not tightened after seeing test coverage.

All selected model settings, ensemble weights, the decision threshold, horizon policies, counting assumptions and interval rules were frozen before the final assessment. No adjustments were made using the test results.

## Assumptions and limitations

- **Two Portuguese hotels, 2015–2017.** Transfer to other properties or current booking behaviour is untested.
- **No booking or guest ID.** Duplicates cannot be verified as true repeats, and the assumption of one room per row directly affects room counts (see the Task 1 sensitivity check).
- **`deposit_type` behaves oddly (F11),** so the models may learn a data artifact. The planned with/without comparison was not run.
- **Only final booking values are stored.** Task 2 is a retrospective benchmark, not a reconstruction of what was known at booking time. Task 1 rebuilds which bookings were open on each date, but many attributes (including deposit and booking-history fields) are still final values. Excluding the known timing-risk columns reduces, but does not remove, this risk.
- **Not fully blind to the test period.** The exploration (Sections 1–10) looked at the full raw dataset. Test performance was reserved for the final assessment only.
- **Task 1 scope.** It covers observed, open reservations. It does not predict future bookings, cancellations already recorded, no-show releases, total occupancy, or rooms left unsold after replacement demand.
- **Task 1 model.** The configuration was reused from Task 2 rather than tuned for Task 1. Model fitting uses unique booking/origin observations, while the historical baseline uses room-night exposure. The one-day zero forecast understates non-zero cancellation counts, and longer horizons keep some signed bias.
- **Uncertainty statements.** Hotel-night errors and repeated booking contributions are correlated over time, so fold SDs and interval coverage are not independent-sample guarantees. The bands come from the earlier validation forecast, before the final refit.
- **Small search budgets.** Conclusions about optimality are limited, and the ensemble's small AUC gains were not tested for statistical significance.
- **No business outcome measured.** There is no price context, event data, capacity or cost information, so revenue recovered and safe overbooking were not assessed. A threshold set in dollars would need the cost of an empty room versus a walked guest. Capacity data, richer historical snapshots and an operational pilot would be needed.
- **Task 3** (cancellation timing) was not implemented.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/FinalDeliverable_Hotel_Cancellation_Forecasting.ipynb
```

Run the notebook top to bottom. It reads `data/raw/hotel_bookings.csv`.

**Runtime and checkpoints.** Sections 15–17 were originally run in Google Colab, saving checkpoints to Google Drive. Outside Colab they fall back to a local folder, `reports/fine_tuning/` (set the `FT_RESULTS_DIR` environment variable to choose another location). Without saved checkpoints, every search and fit is recomputed.

**What has been tested.** A full top-to-bottom run of the final notebook outside Colab, without any Drive checkpoints (Linux, with `FT_RESULTS_DIR` set), completed without errors in about 27 minutes. It reproduced the final test results to four decimals; the only visible difference was in one tuning score (Histogram Gradient Boosting, 0.7989 vs 0.7983), due to library versions. The run has not been tested on Windows or macOS. The re-run of Section 12 with seed 67 was done separately; its recorded outputs are in the notebook.

**Package versions.** `requirements.txt` gives minimum versions so that installation works on most machines. The recorded results were produced in a mix of environments:

| Part of the final notebook | Python | pandas | NumPy | scikit-learn | Other |
|---|---|---|---|---|---|
| Sections 1–14 (local run) | 3.12.15 | 2.3.3 | 2.3.5 | 1.9.1 | matplotlib 3.10.0, seaborn 0.13.2 |
| Sections 15–17 (Google Colab) | 3.13.18 | 2.2.3 | 2.1.3 | 1.6.1 | – |
| Verification run of the whole notebook | 3.13.16 | 3.0.5 | 2.5.3 | 1.9.1 | SciPy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0, threadpoolctl 3.7.0 |

To reproduce the numbers as closely as possible, install the versions from the verification run. `joblib` and `threadpoolctl` are installed automatically with scikit-learn. Saved `.joblib` checkpoints should be reloaded with the same scikit-learn version that created them.

## References

- António, N., de Almeida, A., & Nunes, L. (2017). Predicting hotel bookings cancellation with a machine learning classification model. *16th IEEE International Conference on Machine Learning and Applications (ICMLA)*. https://repositorio-iul.iscte.pt/handle/10071/23459
- António, N., de Almeida, A., & Nunes, L. (2017). Predicting hotel booking cancellations to decrease uncertainty and increase revenue. *Tourism & Management Studies*, 13(2). https://repositorio-iul.iscte.pt/handle/10071/13341
- VentureBeat (2013). HotelTonight raises $45M to spread same-day booking around the world. https://venturebeat.com/business/hoteltonight-raises-45m-to-spread-same-day-booking-around-the-world
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.), Appendix B: Machine Learning Project Checklist.
