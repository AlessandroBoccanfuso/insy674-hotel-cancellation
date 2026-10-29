# INSY 674 Group Project: Hotel Booking Cancellation

McGill Desautels MMA, INSY 674, Fall 2026. 
Dataset: Kaggle *Hotel booking demand* (119,390 bookings, 2 hotels in Portugal, Jul 2015 to Aug 2017).

This README is the documentation required by the assignment rubric. It follows the five checklist sections (Géron, Appendix B). Each section says who owns it and links to the code.

| Section | Owner | Status | Code |
|---|---|---|---|
| 1. Frame the Problem and Look at the Big Picture | Eglantine Dupuy | **done, ready for team review** | this README |
| 2. Explore the Data | Alessandro Boccanfuso| **done, ready for team review** | `notebooks/01_explore_data.ipynb` |
| 3. Prepare the Data | Asad Nadeem | done | `notebooks/03_feature_engineering_and_model_shortlist.ipynb_` |
| 4. Measure and Compare Performance | Logan Jiang, Dario Guo | done | `notebooks/03_Model Cross-Validation` |
| 5. Fine-Tune the System | _TBD_ | not started | `notebooks/04_...` |

## Repository layout

```
data/raw/hotel_bookings.csv   original Kaggle file, never edited
data/README.md                data rules
notebooks/                    one numbered notebook per checklist section
reports/figures/              charts saved by the notebooks, reused in the slides
docs/data_dictionary.md       every attribute: type, noise, distribution, usefulness
requirements.txt              Python packages
```

### Team rules (to avoid merge pain)

- **One notebook per person.** Only edit your own notebook. If you spot a problem in someone else's, leave them a comment instead of changing it.
- **Keep `main` working.** `main` is the official version of the project. It should always run from top to bottom without errors.
- **Use a branch for your changes.** A branch is your own copy to work in, so you can't break the official version. To make one, the git command is `git checkout -b yourname/what-youre-doing`.
- **When you're done, open a pull request.** This is a request on GitHub that says "I'm finished, please check this and add it to `main`." A teammate reviews it and clicks Merge.
- **Use the same random seed** (`random_state=67`) so everyone gets the same results.

## 1. Frame the Problem and Look at the Big Picture

### Our objective

Hotel rooms are extremely perishable: a room that goes unsold on a given night has zero value forever. Cancellations and no-shows free up rooms the hotel can no longer sell through its usual channels, and the hotel loses that revenue.

Our target customers are hotels and other agencies selling stays. Our goal is to help them recover that revenue without risking double bookings or undercutting their own full-price sales. We will use a dataset of 119,390 bookings across two hotels to build a model that:

- forecasts the rooms freed up per hotel per night,
- predicts whether a booking will be cancelled, and
- predicts when that cancellation will happen.

Our model could in turn be used to develop a "Too Good To Go" solution for hotels:

- **Revenue manager dashboard:** expected empty rooms per night with a range (e.g., "likely 8–12 rooms free on Nov 14").
- **Booking-level risk flags:** which reservations are likely to cancel, so the hotel can confirm with the guest, ask for a deposit, or plan around it. Prior research found that bookings contacted by the hotel cancelled less, so outreach is itself a lever.
- **Release recommendation:** when to push a room to our channel, at roughly what discount, and through which channel.
- **Donation tier:** a possible CSR or tax angle for the hotel. When we can confidently predict that a room truly won't sell, we suggest it goes to a fallback tier with a social goal, such as shelters, NGOs, people in need and government partnerships.

### Current solutions and workarounds

- **Revenue management systems (RMS):** already forecast demand and set prices.
- **Hotel policies:** to mitigate cancellations, hotels implement rigid cancellation policies and overbooking tactics, which can hurt revenue and reputation.
- **Last-minute marketplaces:** HotelTonight presents a selection of unsold rooms at steep discounts every day at noon local time. A few other solutions have entered the market.
- **Academic ML work (same dataset authors):** a prototype deployed in two hotels trains a classifier daily to flag "likely to cancel" bookings and compute net demand. Related work reports cancellation prediction accuracy above 90%.

**Our angle:** hotels rarely discount unsold rooms because they fear cannibalizing loyal guests who paid the full price. A forecast tied to a discreet resale channel may address that, but currently no solution does both forecasting and selling. Our solution could fill that gap.

### How will we frame the problem?

**Supervised learning**, with three tasks:

1. **Task 1 (main deliverable):** regression / time series. Forecast rooms free per hotel per night, built by aggregating bookings.
2. **Task 2 (feeds Task 1):** binary classification on `is_canceled`.
3. **Task 3 (potential addition):** predict when a cancellation happens (days before arrival), using survival analysis or regression on cancelled bookings.

Design choices:

- **Offline / batch learning** to start (static historical data). In production, retrain on a schedule (e.g., daily).
- **Several lead times** (e.g., 30, 14, 7 and 1 days out), since hotels decide at different points.
- **Leakage:** drop `reservation_status` and `reservation_status_date`, because they reveal the outcome for Task 1 and 2. If Task 3 is undertook it will be added back to be able to predict time of cancelation. 
- **Split by time, not randomly:** train on earlier arrivals and test on later ones. A random split leaks seasonality and inflates scores.

### How will performance be measured?

- **Forecast:** MAE on rooms free per night, reported by lead time (accuracy at 7–14 days matters more than at 200 days).
- **Prediction interval coverage:** does "8–12 rooms" contain the truth about as often as claimed?
- **Classifier:** ROC-AUC, plus precision and recall, with emphasis on precision (falsely predicting a free room leads to double bookings). Not plain accuracy: predicting "never cancels" already scores ≈ 63%.
- **Calibration** (Brier score, reliability plot), since pricing decisions use the probability itself.
- **Business metric:** revenue recovered per month vs. the hotel's current approach, net of discounts given and any walked guests.

### Is the performance measure aligned with the objective?

Largely, since the hotel acts on per-night counts, which is what we measure. Some gaps remain:

- **Asymmetric costs:** overestimating free rooms (double booking) likely costs more than underestimating (missed resale). We will consider an asymmetric loss (e.g., quantile forecasting aimed slightly low) or a cost-based threshold.
- **Timing:** a correct prediction 3 weeks out is worth more than one the day before. Evaluating by lead time partly addresses this.
- **Model accuracy ≠ business value:** high scores don't prove hotels will act on recommendations. That needs a pilot or A/B test.


## 2. Explore the Data
**Where to find the details**
- `notebooks/01_explore_data.ipynb`: the full analysis, with charts and a written takeaway under each one. Each finding below carries a label (F3, F6...) that matches a takeaway in the notebook.
- `docs/data_dictionary.md`: a table describing every column in the dataset.


1. `reservation_status` and `reservation_status_date` are both dropped to prevent leakage because they contain what we are trying to predict (cancellation). For example, `reservation_status` spells out canceled or check-out. `reservation_status_date` is still useful for building the Task 3 target (days between the cancellation date and arrival), but never as an input feature.  
2. 26.8% of rows are exact duplicates, so we remove them before splitting the data into a test and training set. We do this because some of the duplicates could fall into both the training and test set, which would show the model the correct answer... The duplicates tend to cancel more, which would also inflate the predicted cancel rate. Caveat: the forecast task counts rooms per night, and if these rows are really separate bookings (e.g. several identical rooms booked together), dropping them would undercount demand. There is no booking ID, so we cannot tell (see decision 2). 
3. Lead time is the cleanest driver: 10% cancel when booked within a week, 68% when booked over a year ahead.
4. Groups (61%) and Portuguese guests (57%) cancel most; Direct (15%) and Corporate (19%) least.
5. `deposit_type = Non Refund` cancels 99%, which goes against common sense. People who lose money by cancelling should cancel less. We should look at results with and without it (F11). It could be an error or a problem with how it is recorded. 
6. Several columns are only known after booking (parking, assigned room, booking changes, special requests). If we want to predict at the moment of booking, using them is cheating. The Frame the Problem section forecasts at several lead times (30, 14, 7 and 1 days out), and the file only stores each booking's final values, so we cannot know what these columns looked like 30 days before arrival. Parking and assigned room should be excluded at every horizon; booking changes and special requests should be excluded or handled with care (see the timing table in notebook section 5).  
7. Otherwise clean: 4 columns with missing values (company, agent, country, children), and for `agent`/`company` missing means "none" which is real data.

**Open decisions for the team**

| # | Decision | Status and recommendation |
|---|---|---|
| 1 | Prediction moment | **Settled in Frame the Problem:** forecasts at 30, 14, 7 and 1 days out. Features must be limited to what would be known at each horizon (see finding 6). |
| 2 | Drop exact duplicates? | **Open.** Dropping them protects the classifier from train/test leakage, but may undercount rooms in the per-night forecast. Recommend: drop for the classifier, run the forecast both ways, then decide. |
| 3 | Keep `deposit_type`? | Keep, report with and without, list as limitation. |
| 4 | Random vs time-based split | **Settled in Frame the Problem:** split by time (train on earlier arrivals, test on later). Data covers Jul 2015 to Aug 2017. Use time-ordered cross-validation (not random folds) inside the training period. |
| 5 | Metric | **Settled in Frame the Problem:** MAE by lead time and interval coverage for the forecast; ROC-AUC, precision (emphasised) and calibration for the classifier; revenue recovered as the business metric. Baseline: "never cancels" scores 63% (72.5% once duplicates are removed). |

## 3. Prepare the Data
We prepared the data using the transformation ideas in notebook Section 9 and implemented them in Section 11. From 119,390 original records, we removed 31,994 exact duplicate rows, 166 records with no guests, and one record with negative average daily rate (ADR). This left 87,229 bookings, with a cancellation rate of 27.5%. Zero-ADR and zero-night bookings were retained, with a separate flag identifying zero-night stays. The target was is_canceled, where 1 indicates cancellation and 0 indicates no cancellation.

We sorted the data by arrival date and allocated the earliest 70% of unique arrival dates to training, the next 15% to validation, and the remaining 15% to testing. These percentages refer to dates rather than booking counts.
![alt text](image.png)
After record-level cleaning and splitting, feature construction and preprocessing were implemented through BookingFeatureEngineer, ColumnTransformer, and a scikit-learn Pipeline. The main transformations were:

Feature selection: We excluded reservation_status and reservation_status_date, which reveal the outcome, together with assigned_room_type, booking_changes, days_in_waiting_list, required_car_parking_spaces, and total_of_special_requests, whose timing creates a leakage risk for booking-time prediction. Raw company IDs were replaced by has_company; distribution_channel was also excluded.

Missing and categorical values: Missing children counts were filled with zero, missing countries with UNKNOWN, and missing agents with NO_AGENT. We created has_agent and has_company, merged undefined meal categories into SC, and mapped undefined market segments to UNKNOWN.

Feature engineering: We created total nights, total guests, family and baby indicators, weekend-stay ratio, ADR per guest, booking-history indicators and historical cancellation rate. We also added lead-time buckets, log transformations of lead time, ADR and total nights, arrival weekday and weekend indicators, cyclical month features, and hotel–market-segment interactions. Historical variables were assumed to describe bookings preceding the prediction time.

Learned preprocessing: Selected numeric features were capped at their training-set 99.5th percentiles, median-imputed and standardized. Categorical features were one-hot encoded, grouping categories occurring fewer than 100 times in the fitting data into an infrequent category.

All learned cutoffs, medians, scaling parameters and category groupings were fitted on training data only. During cross-validation, the complete preprocessing-and-model pipeline was cloned and refitted within each training fold. Held-out data used the fitted transformations without refitting. The baseline pipeline fitted on the full training period produced 171 encoded features. deposit_type was retained in the reported experiments.

## 4. Measure and Compare Performance
We compared eight algorithms covering linear, margin-based, tree-ensemble, probabilistic and neural-network approaches. All experiments used the same five chronological, expanding-window folds within the training period, with random_state=67 where applicable. Each fold trained on earlier arrival dates and evaluated on the following period, keeping bookings with the same arrival date together. We computed the mean and standard deviation of ROC-AUC, precision, recall, F1 and Brier score across folds. ROC-AUC was the main ranking metric; Brier score was unavailable for Linear SVM because its decision scores were not probabilities. The final test set was reserved for later evaluation.

For each algorithm, we evaluated three feature versions using unchanged model settings: V0 used the baseline features; V1 removed is_domestic, which overlaps with country; and V2 added estimated_stay_value = adr × total_nights to V1. This produced 24 model–feature combinations. Improvements were not consistent across algorithms: Extra Trees improved from 0.8111 to 0.8170 mean AUC between V0 and V2, while MLP retained its highest mean AUC with V0.

The highest-mean-AUC representative for each algorithm is shown below. FP and FN rates come from the separate validation set, using a probability threshold of 0.5 where available and the standard decision boundary for Linear SVM.
![alt text](image-1.png)

We assessed important variables using validation-set permutation importance, measured as the ROC-AUC decrease after shuffling each engineered feature, repeated five times. 

Removing is_domestic often increased the measured importance of country, consistent with their overlapping information. The added stay-value feature did not provide a consistent improvement across models.

Confusion matrices, subgroup comparisons by hotel and market segment, and confidently incorrect predictions revealed different error patterns. Extra Trees and Random Forest generated few false alarms but missed many cancellations. MLP and Histogram Gradient Boosting reduced missed cancellations at the cost of more false alarms. Gaussian Naive Bayes detected more cancellations but incorrectly flagged 35.53% of non-canceled validation bookings. Some missed cancellations involved short lead times and no previous cancellations, while some false alarms involved long lead times or a cancellation history. Resort Hotel also showed higher missed-cancellation rates in several comparisons. Guest identifiers and prior behavior, detailed cancellation terms, payment information and relative market prices, where available at prediction time, could help distinguish these cases.

We shortlisted Extra Trees V2, MLP V0, Logistic Regression V1 and Histogram Gradient Boosting V2. Extra Trees had the highest mean CV AUC; MLP combined a strong AUC with a different error pattern; LR provided an interpretable linear probability model; and Histogram Gradient Boosting offered a different FP–FN trade-off. Validation error-set Jaccard overlap was 56.2% between Extra Trees and MLP, and 61.0% between Extra Trees and Histogram Gradient Boosting. By comparison, Random Forest overlapped 89.2% with Extra Trees, while Linear SVM overlapped 91.6% with LR. SGD Logistic offered no clear performance advantage, and Gaussian Naive Bayes had weaker AUC and substantially more false alarms. Selection therefore considered performance, interpretability and error diversity together. Variation across time folds remained substantial, so small mean-score differences were interpreted cautiously. These four models were retained for subsequent tuning and final testing.
## 5. Fine-Tune the System
5. Fine-Tune the System
Implementation: 05_Fine_Tune_and_Forecast_Antonin.ipynb

5.1 Scope and evaluation design

This section continues from the team's preparation, chronological model comparison, and four-model shortlist. Task 2 predicts the binary is_canceled outcome, which includes both explicit cancellations and no-shows. The additional Task 1 implementation forecasts room-nights released by subsequent explicit pre-arrival cancellations among observed reservations.

For Task 2, we preserved the team's cleaning rules, feature versions, and chronological partitions: 56,132 training bookings, 14,009 validation bookings, and 17,088 test bookings. Five expanding-window folds were constructed within the training period using unique arrival dates. Each fold trained on earlier arrivals and evaluated on a subsequent period. Feature engineering, imputation, outlier capping, category encoding, and scaling were fitted within each fold's pipeline.

Mean CV ROC-AUC was the model-selection criterion. We also recorded fold standard deviation, precision, recall, F1, and Brier score. Classification metrics initially used the default 0.5 decision boundary. CV means are averages of the five fold metrics; the accompanying standard deviations describe variation across periods, not confidence intervals.

5.2 Randomized tuning of models and transformations

We used randomized parameter sampling with the existing five chronological folds. To make the experiments reproducible and recoverable, we used random seed 67 and saved results after every completed fold. An initial broader Extra Trees search was interrupted because individual fits were too expensive; its incomplete results were not used to select a winner. The completed search used a restricted computational budget.

Shortlisted configuration

Randomized alternatives

Untuned reference also evaluated?

Original mean CV AUC

Retained mean CV AUC

Extra Trees V2

6

Yes

0.8170

0.8170

Logistic Regression V1

3

Yes

0.7984

0.7993

Histogram Gradient Boosting V2

2

Yes

0.7932

0.7983

MLP V0

2

Yes

0.8130

0.8130

Feature-version labels refer to the existing feature sets. V1 removes is_domestic while retaining country; V2 additionally introduces estimated stay value (adr × total_nights).

The Extra Trees search varied depth, minimum leaf size, and class weighting, while fixing 90 trees and square-root feature sampling to control runtime. It jointly explored no outlier capping versus 99th- and 99.5th-percentile capping, median versus mean numeric imputation, and rare-category minimum frequencies. These imputation choices apply to missing values reaching the numeric imputers; they do not override domain-based replacements already performed inside the feature engineer.

Logistic regression tuning explored regularization strength, class weighting, and rare-category grouping. Boosting tuning explored learning rate, iteration count, leaf count, minimum leaf size, and L2 regularization. MLP tuning explored hidden-layer size, weight regularization, and initial learning rate, retaining the existing 90-iteration maximum and early stopping.

The retained configurations were:

Extra Trees V2: original settings: 90 trees, maximum depth 18, minimum leaf size 4, square-root feature sampling, and no class weighting. None of the six completed alternatives exceeded its original mean CV AUC.

Logistic Regression V1: C = 0.4340645213116581, no class weighting, and rare-category minimum frequency 100. The AUC improvement was small.

Histogram Gradient Boosting V2: learning rate 0.05, 60 maximum iterations, 15 maximum leaf nodes, minimum leaf size 50, and L2 regularization 0.

MLP V0: one hidden layer of 48 neurons, alpha = 0.01142030649919016, and initial learning rate 0.001. Its observed AUC improvement was negligible at four-decimal precision.

These were bounded searches, not evidence of exhaustive optimization. We retained original settings whenever the completed alternatives did not improve the declared criterion.

5.3 Ensemble comparison

We evaluated two predefined equal-weight soft-voting ensembles using cancellation probabilities for the same held-out bookings in each chronological fold. Constituent pipelines were fitted independently within each fold. We compared the two ensembles with the retained individual configurations.

System

Mean CV ROC-AUC

Fold SD

Precision

Recall

F1

Brier

Extra Trees + MLP

0.8214

0.0737

0.7440

0.2743

0.3796

0.1408

All four models

0.8203

0.0743

0.7431

0.2640

0.3662

0.1420

Extra Trees V2

0.8170

0.0778

0.7634

0.2642

0.3625

0.1383

MLP V0

0.8130

0.0736

0.7068

0.2919

0.3971

0.1492

Logistic Regression V1

0.7993

0.0863

0.6695

0.2693

0.3671

0.1478

Histogram Gradient Boosting V2

0.7983

0.0836

0.7371

0.2670

0.3590

0.1490

We selected the Extra Trees–MLP ensemble by mean CV ROC-AUC. Its 0.0044 improvement over Extra Trees indicates better observed risk ranking, while Extra Trees retained the lowest CV Brier error. The ensemble therefore did not dominate every measure.

5.4 Validation, threshold choice, and final Task 2 evaluation

The selected ensemble was fitted on the original training partition and evaluated on the separate validation period. Validation ROC-AUC was 0.8346 and Brier score was 0.1474. A reliability plot showed imperfect calibration, including underprediction in several middle-to-high probability groups.

Because the project emphasizes avoiding false cancellation flags, we selected a decision threshold by maximizing validation F0.5 over the fixed grid 0.05, 0.10, …, 0.95. F0.5 emphasizes precision but is a development proxy, not an estimated hotel cost function. The selected threshold was 0.45, with the rule predicted probability > 0.45.

After freezing the configurations, equal ensemble weights, and threshold, we refitted both constituent pipelines on all 70,141 development bookings and evaluated the reserved test partition.

System / decision rule

Test ROC-AUC

Brier

Precision

Recall

F1

F0.5

Development-rate probability baseline

0.5000

0.2344

0.0000

0.0000

0.0000

0.0000

Original Extra Trees configuration, threshold 0.50

0.7961

0.1734

0.6763

0.2933

0.4091

0.5362

Selected ensemble, threshold 0.50

0.8040

0.1703

0.6391

0.3966

0.4894

0.5694

Final ensemble, threshold 0.45

0.8040

0.1703

0.6090

0.5196

0.5607

0.5887

The final operating point produced 9,185 true negatives, 1,977 false positives, 2,847 false negatives, and 3,079 true positives. Relative to the ensemble at threshold 0.50, it detected 729 additional cancellations/no-shows with 650 additional false positives. Threshold changes affect decisions, not ROC-AUC or Brier score.

The final ensemble improved observed test AUC and probability error relative to the original Extra Trees configuration. However, precision of 60.9% means that approximately 39.1% of cancellation flags were false alarms. The system supports prioritizing investigation or confirmation; the results do not justify automatically releasing flagged rooms for resale. No model or threshold changes were made in response to these test results.

5.5 Task 1 extension: cancellation-related room-night forecasting

Forecast target. We estimated subsequent explicit pre-arrival cancellations among observed reservations still open at a historical prediction date and covering a particular hotel-night. This differs from total hotel availability: capacity, replacement bookings, and complete inventory were unavailable. No-shows were excluded from this target.

Snapshot construction. Estimated booking date was reconstructed as arrival date minus lead time. Stays were expanded into nights, excluding departure day. A booking-night was eligible if it had already been booked, arrival was still in the future, and its terminal status had not yet been recorded at the end of the prediction date. Cancellations already known at that date were excluded. Dates were treated as end-of-day cutoffs because exact timestamps were unavailable.

Time separation. Initial training nights ran from July 31, 2015 through January 5, 2017, using only terminal outcomes recorded by that fitting cutoff. The common validation-night window was February 5–May 4, 2017, ensuring even 30-day forecast origins followed the training cutoff. Initial training contained 382,025 booking-night-horizon rows, reduced to 368,660 distinct booking-row/origin observations for model fitting.

Conditional model. We reused the retained logistic-regression configuration and fitted fresh weights to the explicit-cancellation snapshot target. We added days until arrival and days since booking. Outcome/status fields and other previously excluded timing-risk columns were withheld from the predictors. This was one fixed conditional-model fit, not another Task 1 hyperparameter search.

The conditional probabilities were summed by hotel, night, and horizon. We compared this forecast with training-period cancellation rates by hotel/horizon and a zero-cancellation forecast. Threshold 0.45 was not applied to the count forecast.

Validation policy. We selected the lowest-MAE method per horizon, with RMSE as the tie-breaker. The zero forecast was selected at one day; the snapshot model was selected at seven, fourteen, and thirty days. The zero forecast is a conservative count prediction selected for MAE, not evidence that cancellations are impossible.

Duplicate sensitivity. We held the risk model, historical rates, and selected methods fixed while counting identical records separately. Validation MAE increased from 0.3090 to 0.4045 at one day, 2.1291 to 2.6552 at seven days, 2.4066 to 3.0429 at fourteen days, and 3.6254 to 4.4869 at thirty days. This shows sensitivity to the counting assumption; it does not establish whether duplicate rows are database repeats or genuine separate reservations.

Final refit and evaluation. We refitted the conditional model using 451,745 distinct booking-row/origin observations with outcomes recorded by May 4, 2017. The selected methods were unchanged. Test nights ran from June 4–August 31, 2017; even the earliest 30-day test origin followed the final fitting cutoff. Evaluation covered 178 hotel-nights per horizon, or 712 hotel-night-horizon records.

Horizon

Frozen method

Test MAE

Historical-rate MAE

Zero-forecast MAE

RMSE

Bias

1 day

Zero forecast

0.2809

0.4652

0.2809

0.5996

−0.2809

7 days

Snapshot model

1.7456

3.2066

2.4045

2.1376

+1.1221

14 days

Snapshot model

2.1857

4.5971

6.6236

2.8446

−0.9854

30 days

Snapshot model

3.9414

6.0855

13.2921

5.0549

+1.0784

Bias is forecast minus actual. Averaging equally across the four horizons, the frozen policy's test MAE was approximately 2.0384, compared with 3.5886 for historical rates: approximately 43.2% lower error. These averages combine equally sized horizon groups and are calculated from the displayed rounded results.

Uncertainty bands. We used hotel/horizon-specific 5th and 95th percentiles of validation residuals (actual minus selected forecast), rounded count bounds outward, and clipped lower bounds at zero. These are nominal 90% empirical error bands, not formal coverage guarantees.

Horizon

Test coverage

Mean band width, room-nights

1 day

98.31%

1.5000

7 days

99.44%

5.9607

14 days

94.38%

9.2360

30 days

93.82%

15.6404

Coverage exceeded 90% in this test period, with relatively wide bands, particularly at thirty days. We did not tighten the bands after inspecting test coverage.

5.6 Assumptions and limitations

The data represent two Portuguese hotels in 2015–2017; transfer to other properties or current booking behavior is untested.

The team's earlier exploration examined the full raw dataset. Final test performance was reserved for the final assessment, but the project was not fully blind to the test-period data throughout every earlier exploratory stage.

Task 2 follows the team's arrival-date holdout protocol. It is a retrospective classification benchmark, not a fully reconstructed booking-time deployment backtest. Task 1 additionally enforces fitting cutoffs, outcome availability, and later forecast origins.

Snapshot reconstruction remains partial: many attributes are final stored values. Excluding known timing-risk features reduces leakage risk but does not establish that every retained attribute, including deposit and historical-booking fields, was recorded identically at the historical prediction moment.

There is no genuine booking identifier. Duplicate interpretation and the assumption of one room per row directly affect room counts.

Task 1 concerns observed, open pre-arrival reservations. It does not predict future bookings, cancellations already recorded, no-show releases, total occupancy, or rooms remaining unsold after replacement demand.

The conditional forecast model and its historical baseline have different weighting: model fitting uses unique booking/origin observations, while historical rates use room-night exposure. The model is a practical extension whose configuration was reused from Task 2 rather than separately optimized.

The one-day zero forecast minimizes validation MAE but systematically understates nonzero cancellation counts. Longer-horizon forecasts also retain signed bias.

Hotel-night errors and repeated booking contributions are temporally correlated. Fold SDs and empirical interval coverage should not be interpreted as independent-sample statistical guarantees.

Interval residuals came from the earlier validation forecast, before the final refit. The resulting bands have no formal guarantee under changing model fits or booking patterns.

Small search budgets limit conclusions about optimality. The ensemble's small AUC gains were observed, not demonstrated statistically significant.

Cancellation prediction does not demonstrate revenue recovered or safe overbooking. Capacity information, richer historical snapshots, and an operational pilot would be required to assess those business outcomes.

Task 3, cancellation timing prediction, was not implemented in this contribution.

All selected model settings, ensemble weights, decision thresholds, horizon policies, counting assumptions, and interval rules were frozen before final performance assessment. No adjustments were made using the final test results.
## Assumptions and limitations (running list for the slides)
- Two hotels, both in Portugal, 2015 to 2017: results may not transfer.
- No booking or guest ID, so duplicates cannot be verified as true repeats (dropping real ones would undercount rooms in the per-night forecast).
- Only final booking values are stored, so we cannot reconstruct what was known 30, 14, 7 or 1 days before arrival; columns recorded late (booking changes, special requests) risk leaking future information.
- `deposit_type` behaves oddly (F11), so the model may learn a data artifact.
- No price context, events, or capacity data; value estimate will rely on assumed room-night costs.

## Running it
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_explore_data.ipynb
```
