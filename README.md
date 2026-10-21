# INSY 674 Group Project: Hotel Booking Cancellation

McGill Desautels MMA, INSY 674, Fall 2026. 
Dataset: Kaggle *Hotel booking demand* (119,390 bookings, 2 hotels in Portugal, Jul 2015 to Aug 2017).

This README is the documentation required by the assignment rubric. It follows the five checklist sections (Géron, Appendix B). Each section says who owns it and links to the code.

| Section | Owner | Status | Code |
|---|---|---|---|
| 1. Frame the Problem and Look at the Big Picture | Eglantine Dupuis | not started | this README |
| 2. Explore the Data | Alessandro Boccanfuso| **done, ready for team review** | `notebooks/01_explore_data.ipynb` |
| 3. Prepare the Data | _TBD_ | not started | `notebooks/02_...` |
| 4. Measure and Compare Performance | _TBD_ | not started | `notebooks/03_...` |
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
_To be written by the team. Checklist prompts: objective; how the solution will be used; current workarounds (e.g. blanket overbooking, deposits); supervised vs unsupervised, online vs offline; performance measure and why it matches the business objective._

**Inputs from the exploration that affect the framing** (see notebook, findings F5, F6, F9):
- Natural target: `is_canceled` (binary classification), 37.0% of the bookings are cancelled. (27.5% after removing duplicates).
- The team must decide **when the prediction is made** (at booking vs shortly before arrival). It decides which columns are legal features.
- Accuracy alone is a weak metric (always predicting "no cancel" scores 63%). Use precision/recall/F1 or PR-AUC tied to the cost of an empty room vs an overbooked guest.

## 2. Explore the Data
**Where to find the details**
- `notebooks/01_explore_data.ipynb`: the full analysis, with charts and a written takeaway under each one. Each finding below carries a label (F3, F6...) that matches a takeaway in the notebook.
- `docs/data_dictionary.md`: a table describing every column in the dataset.


1. `reservation_status` and `reservation_status_date` are both dropped to prevent leakage because they contain what we are trying to predict (cancellation). For example, `reservation_status` spells out canceled or check-out.  
2. 26.8% of rows are exact duplicates, so we remove them before splitting the data into a test and training set. We do this because some of the duplicates coult fall into both the training and test set, which would show the model the correct answer... The duplicates tend to cancel more, which would also inflate the predicted cancel rate. 
3. Lead time is the cleanest driver: 10% cancel when booked within a week, 68% when booked over a year ahead.
4. Groups (61%) and Portuguese guests (57%) cancel most; Direct (15%) and Corporate (19%) least.
5. `deposit_type = Non Refund` cancels 99%**, which goes against common sense. People who lose money by cancelling should cancel less. We should look at results with and without it (F11). It could be an error or a problem with how it is recorded. 
6. Several columns are only known after booking (parking, assigned room, booking changes, special requests). If we want to predict at the moment of booking, using them is cheating. Same point as **when the prediction is made** from the frame the problem section.  
7. Otherwise clean: 4 columns with missing values (company, agent, country, children), and for `agent`/`company` missing means "none" which is real data.

**Open decisions for the team**

| # | Decision | Recommendation |
|---|---|---|
| 1 | Prediction moment (at booking vs before arrival) | Choose in section 1; drives feature list |
| 2 | Drop exact duplicates? | Yes, before splitting; run one experiment keeping them |
| 3 | Keep `deposit_type`? | Keep, report with and without, list as limitation |
| 4 | Random vs time-based split | Random split in CV; time-based hold-out for the final test set |
| 5 | Metric | PR-AUC / F1 / recall at a target precision |

## 3. Prepare the Data
_To be written. See the transformation table in notebook section 9. Everything must live in a scikit-learn `Pipeline` fit on the training data only._

## 4. Measure and Compare Performance
_To be written. N-fold CV with mean and std, several model families, error analysis, shortlist of 3 to 5._

## 5. Fine-Tune the System
_To be written. Randomized search with CV, ensembles, one final evaluation on the untouched test set._

## Assumptions and limitations (running list for the slides)
- Two hotels, both in Portugal, 2015 to 2017: results may not transfer.
- No booking or guest ID, so duplicates cannot be verified as true repeats.
- `deposit_type` behaves oddly (F11), so the model may learn a data artifact.
- No price context, events, or capacity data; value estimate will rely on assumed room-night costs.

## Running it
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_explore_data.ipynb
```
