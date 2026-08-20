---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---
## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff only — remove before sharing with students)*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟢 | The project utilizes Python-based tools and libraries for ML, which aligns well with students' existing skills from ML Foundations. |
| Data Readiness | 🟢 | The dataset is under 1GB, making it feasible for students to utilize without extensive cleaning, allowing focus on modeling and analysis. |
| Resource Check | 🟢 | The use of Google Colab's free tier makes the project accessible regarding hardware requirements, with no proprietary tools involved. |

**Student Fit Score:** 7/10  
**Technical Depth Score:** 8/10  
**Overall Recommendation:** APPROVE

**Advisor Feedback Draft:**
The project presents an engaging application of machine learning to real-world health data, suitable for students. However, focus on the imbalanced dataset will demand extra attention in teaching and may benefit from a preliminary session on handling such datasets. Consider integrating targeted learning modules to address this aspect while keeping students aligned with the data and ML techniques.

---

# Predicting Heart Disease Risk from Health Survey Indicators

**Company / Org:** Accenture  
**Challenge Advisor:** Joseph Chiasson, josephlchiasson@outlook.com   
**AI Studio Coach:** Swagath Babu, swagath.babu@breakthroughtech.org    
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Accenture

Accenture is a global professional services company specializing in digital, cloud, and security services, and consulting. We help clients build their digital capabilities to improve productivity and achieve tangible results.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use anonymized  national health-survey data from the CDC's Behavioral Risk Factor Surveillance System (BRFSS 2022) and supervised machine learning techniques (e.g. logistic regression, tree-based ensembles, and gradient boosting) to build an ML model that predicts whether an individual has a history of coronary heart disease or heart attack from lifestyle, demographic, and self-reported clinical risk factors.   What makes this challenge project unique is the fact that it is an example of an imbalanced dataset, in which only 5-6% of the roughly 445k total records are of positive prevalence/class.

### Success Criteria
With imbalanced datasets, the traditional ML classification metrics such as Accuracy can be misleading. In this case, for example, a 94-95% accuracy could in theory fail to detect any of the positive class occurrences. Therefore, alternate metrics such as PR-AUC, F1, ROC-AUC should be used. Students should also consider how effective (or biased) the ML model across certain demographic subgroups, such as race/ethnicity, age sex, and income.

### Stretch Goals
A stretch goal for this project would be to attempt the ML prediction with a neural network implementation versus classic ML algorithms.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | EDA Notebook | • **Predictor variable/feature typing:** Explicitly characterize each variable (numerical, categorical) and document encoding decisions such as one-hot, ordinal, etc.<br>• **Missing data:** Identify missing data and document how your team handled it.<br>• **Interrelationships:** Examine the correlations between any variables and the potential implications.<br>• **Exploratory Data Analysis:** Analyze distributions of all features, including bivariate analysis.<br>• **Split data:** Divide data into training/validation/test with a fixed random seed, being sure that the positive rate is maintained across the splits.<br>• **DELIVERABLE:** An EDA notebook with documented findings and a written summary of conclusions. |
| October | Modeling Notebook | • **Feature Engineering:** Consider any numerical features where scaling/normalization might be required. If you apply them, be sure that the transformation technique is derived only from the training data to prevent look-ahead bias.<br>• **Modeling:** Train 3 or more different classification algorithms, each with cross validation and hyperparameter tuning. Likely algorithm candidates include:<br>&nbsp;&nbsp;&nbsp;&nbsp;◦ Logistic Regression (with L1 & L2 regularization)<br>&nbsp;&nbsp;&nbsp;&nbsp;◦ Random Forest<br>&nbsp;&nbsp;&nbsp;&nbsp;◦ K Nearest Neighbors<br>&nbsp;&nbsp;&nbsp;&nbsp;◦ Gradient Boosting (XGBoost or LightGBM)<br>• **Experiment Tracking:** Maintain a results table that compares the performance of each model at various settings.<br>• **Optional/Advanced:** Consider oversampling techniques on the minority positive class, such as random oversampling or SMOTE.<br>• **DELIVERABLE:** A modeling notebook with reproducible training runs and a comparison table of the results. |
| November | Final Model and Evaluation Report | • **Evaluation:** Comprehensive evaluation on the held-out test data set — PR-AUC, F1, ROC-AUC, Precision, and Recall at multiple operating thresholds, plus a confusion matrix at the chosen operating threshold.<br>• **Interpretability:** SHAP analysis on the best model; identify top drivers of predicted risk.<br>• **Subgroup Fairness Analysis:** Consider how model performance differs across key variables such as race/ethnicity, sex, age, income, and US region.<br>• **DELIVERABLE:** Recommend final model, evaluation report, and subgroup fairness analysis. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** Indicators of Heart Disease (2022)   
**Format:** CSV  
**Size:** under 1gb  
**Location:** https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

### Key Details
- Anonymized national health-survey data containing numerical and categorical data.
- Known limitations include dealing with the imbalanced nature of the dataset.
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- PR-AUC, F1 Score, ROC-AUC, Precision, Recall, and confusion matrix.

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- https://machinelearningmastery.com/5-effective-ways-to-handle-imbalanced-data-in-machine-learning/
- https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets

**Technical Tutorials:**
- https://www.geeksforgeeks.org/machine-learning/smote-for-imbalanced-classification-with-python/
- https://arxiv.org/pdf/2310.07917
- https://imbalanced-learn.org/stable/references/over_sampling.html

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
