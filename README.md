# Student Performance

## Summary

We use survey data from two Portuguese secondary schools to predict students' exam scores. 
Multiple models are considered and compared: Multivariate Regression (with Lasso and Ridge Regularization and hyperparameter tuning) and a Random Forest Regressor. Additionally, we quantify the increase in model accuracy due to common feature engineering techniques: binning rare categories and dropping outliers. Based on these results, we present the best performing model and analyze its accuracy.


## Description

We consider student data from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance) (see also the citation in [References](#references)).  The survey data includes student grades, demographics, and social and school related features. It was collected from school reports and questionnaires. We also have the exam grades, called 'G3'. We will train various machine learning models on the student survey data and measure the models' success in predicting the students' G3 grade.

It is common for a school district to desire an 'early alert system' that automatically detects if a student is at risk of failing a course. Ideally, this system would use information obtained early in the semester (or before it has begun), so that the teacher/administration can intervein as early as possible. Often times, midterm exam scores will not yet be available. For this reason, we drop the 'G1' and 'G2' exam scores within the data set and only use student surveys to train the models.

An early alert system of this type, which uses student surveys taken at the beginning of the semester,  has the potential of identifying students in danger of failing the course. In such a case, the teacher has the opportunity to reach out and provide the student with additional resources.

## Files
All code is written in Python (version 3.10.8).

- [ ] Student_Performance.ipynb : Main notebook containing results.
- [ ] student-por.csv : Student survey data.
- [ ]  Misc\Exploratory.ipynb : We perform Exploratory Data Analysis (EDA), using the pandas profiling package and seaborn plots to gain insight into the data.
- [ ]  Misc\Draft.ipynb : As in Exploratory.ipynb, we perform EDA. We also include scatterplots with lines of best fit and drafts of basic models to understand the project outline.

## Data cleaning and feature engineering

### Feature engineering, step 1: Bin rare categories.

We consider any category that contains less than 5% of students a rare category. We find the rare categories and group the data with related categories (i.e. binning) to remove any rare categories. For example, we bin the feature 'age' in the following way:

	# age=19,20,21,22 are rare categories at 4.93 %, 0.92 %, 0.31 %, and 0.15 % 
	representation, respectively.
	# We will group them into a single age '19'.
	# Define the function for binning:

	def get_age(N):    
	    if N<19 :
	        return N	            
	    else:
	        return 19
	        
    # Create a new feature 'age_bin' 
	df_full['age_bin'] = df_full['age'].apply(get_age)


### Feature engineering, step 2: add polynomial features.

The features 'freetime' and 'goout' should have a quadratic fit and not a linear fit. 

The reason behind this decision is that students who have very little free time (freetime = 1) score worse than students with some free time (freetime = 2). That is, 'G3' increases from freetime=1 to freetime=2. However, students with freetime > 2 see a monotonic decrease in their 'G3' score. This behavior can be modelled well by a quadratic fit. 

We demonstrate the behavior by showing the box plot for 'freetime' and 'goout'. We see that the distribution of 'goout' behaves similarly to 'freetime'.

	sns.catplot(data=df_full, x="freetime", y="G3", kind="box")



![freetimeBox](Misc/freetimeBox)



### Preprocessing: one-hot encode the categorical features


	enc_unbin = OneHotEncoder(handle_unknown='ignore',drop='first')

Choose the categorical columns:

	categorical_cols_unbin = [cname for cname in df_unbin.columns if 
			df_unbin[cname].nunique() < 10 and df_unbin[cname].dtype == "object"]

Extract the categorical_cols and apply one-hot:

	df_unbin_onehot = df_full[categorical_cols_unbin].copy()
	enc_unbin.fit(df_unbin_onehot)
	df_unbin_onehot = pd.DataFrame(enc_unbin.transform(df_unbin_onehot).toarray())

Now drop the categorical_cols and add in the onehot encoded:

	df_unbin = df_unbin.drop(categorical_cols_unbin, axis=1).copy()
	df_unbin = df_unbin.merge(df_unbin_onehot, how='left', 
			left_index=True, right_index=True)

### Exploratory data analysis:

We have performed the majority of our EDA in the file Misc/Exploratory.ipynb, wherein we use the pandas profiling package and seaborn plots to gain insight into the data.

As a result of that EDA, we remark that the feature 'failures' is a strong predictor for exam scores 'G3'. A student who has failed at least one course in the past is expected to perform worse than a student who has not failed any courses. We can see this by plotting a line of best fit to the scatterplot of 'failures' against 'G3'.



	sns.catplot(data=df_line_fit_1_drop2, x="failures", y="G3", kind="swarm", s = .9)
	plt.plot(X_test_subset_1, y_pred_subset_1, color="blue", linewidth=2)


![failuresScatter](Misc/failuresScatter)



We can also see this relationship by plotting the correlation matrix as below, and confirm that 'failures' has a high correlation with 'G3'.
	
	corr_matrix_unbin = df_unbin.corr()
	sns.heatmap(corr_matrix_unbin)
	plt.show()


### Feature engineering, step 3: Dropping outliers.

A small number of students did not take exam 'G3' and thus scored a 0. A student may not take the exam for a number of reasons (poor health, family emergency, etc.).

We will consider the effect on our models of dropping the rows where G3=0. We may do this using the following code:

	# Remove the 15 students who did not take the G3 exam. 
	# This includes 15 students, or 2.3% of the total population.

	df_out_bin = df_bin.loc[ df_bin['G3'] != 0 ]
	display(df_out_bin.shape)


## Modeling: Linear Regression vs. Random Forest Regressor
We have 4 datasets: 
- __df_unbin__: Corresponding to the original unbinned data.
- __df_bin__: Corresponding to the binned data.
- __df_out_unbin__: Same as df_unbin except dropping the students who did not take the exam.
- __df_out_bin__: Same as df_bin except dropping the students who did not take the exam.

We will consider two models applied to the above data sets: 
- Linear Regression (with Lasso and Ridge regularization) 
- Random Forest Regression

We compare the performance of these models against the benchmarks:
- Bench 1: Predict every student scored the average score.
- Bench 2: Linear regression using one feature, 'failures'. 
    - We choose to use 'failures' because it is the feature that has the highest correlation with 'G3': correl = -0.39

We use the Mean Absolute Error (MAE) as our scoring metric. We have the following results:

	Results for df_unbin:
	 
	Bench 1, pred_avg
	     MAE: 2.29
	 
	Bench 2, pred_failures
	     MAE: 2.17
 
	Model 1, lin reg
	     MAE: 1.90
	 
	Model 1a, Lasso
	     MAE: 2.00
	 
	Model 1a tuned, Lasso
	     MAE: 1.93
	     a =  0.05
	 
	Model 1b, Ridge
	     MAE: 1.90
 
	Model 1b tuned, Ridge
	     MAE: 1.93
	     a =  37.00
	 
	Model 2, forest
	     MAE: 1.89
	 
	-----------------------------------

As expected, the benchmarks perform poorly, with error > 2 points (out of 20 total points for the exam).  Linear regression, tuned Lasso, and Random Forest perform the best with MAE ~ 1.9. We see that tuned Ridge performs worse than default Ridge, which means the tuning in this case is overfitting to the validation set. An alternative tuning objective should be considered to enhance generalizability.
	 
	Results for df_bin:
	 
	Bench 1, pred_avg
	     MAE: 2.29
	 
	Bench 2, pred_failures
	     MAE: 2.13
	 
	Model 1, lin reg
	     MAE: 1.88
 
	Model 1a, Lasso
	     MAE: 1.94
	 
	Model 1a tuned, Lasso
	     MAE: 1.90
	     a =  0.06
	 
	Model 1b, Ridge
	     MAE: 1.88
	 
	Model 1b tuned, Ridge
	     MAE: 1.91
	     a =  34.00
	 
	Model 2, forest
	     MAE: 1.87
	 
	-----------------------------------
	 

Binning did improve the scores for each of our models, but only marginally. For example, linear regression only improved 0.02 points. 


	Results for df_out_unbin:
	 
	Bench 1, pred_avg
	     MAE: 2.18
	 
	Bench 2, pred_failures
	     MAE: 2.10
	 
	Model 1, lin reg
	     MAE: 1.66
	 
	Model 1a, Lasso
	     MAE: 1.81
	 
	Model 1a tuned, Lasso
	     MAE: 1.69
	     a =  0.02
	 
	Model 1b, Ridge
	     MAE: 1.66
	 
	Model 1b tuned, Ridge
	     MAE: 1.68
	     a =  24.00
	 
	Model 2, forest
	     MAE: 1.82
	 
	-----------------------------------
	 

We see that dropping outliers (students who did not take the G3 exam) gave a significant improvement to the accuracy of linear regression (from ~1.9 to ~1.66 points). This is to be expected since ordinary least squares is sensitive to outliers. Lasso and Ridge models saw a similar improvement.
The Random Forest model only saw a minor improvement (~0.07 points).

	Results for df_out_bin:
	 
	Bench 1, pred_avg
	     MAE: 2.18
	 
	Bench 2, pred_failures
	     MAE: 2.07
	 
	Model 1, lin reg
	     MAE: 1.66
	 
	Model 1a, Lasso
	     MAE: 1.82
	 
	Model 1a tuned, Lasso
	     MAE: 1.71
	     a =  0.03
	 
	Model 1b, Ridge
	     MAE: 1.66
	 
	Model 1b tuned, Ridge
	     MAE: 1.69
	     a =  21.00
	 
	Model 2, forest
	     MAE: 1.82
	 
	-----------------------------------

We see that the tuned Lasso model is slightly worse than the previous result (1.71 rather than 1.69), however the other models have identical results.

## Conclusion


The Linear Regression model has the best accuracy at MAE = 1.66 . However, we recommend the hyperparameter tuned Lasso regularization model for use as an early alert system. It has similar accuracy (~1.69) and will be more robust to multicollinearity issues. Lasso also performs more reliably than Ridge in this case, as Ridge tends to overfit to the validation set. We did not see much improvement by binning the rare categories, but we saw a large improvement by dropping the outliers (students who did not take the 'G3' exam). 
 

Since the exam is out of 20 points, an error of 1.69 points is ~ 8.5 % accuracy. To put the result in context, if we use a letter grade distribution (e.g. we assign the letter grade A if student scored a 90-100%, a B if they scored 80-90%, etc...), we may interpret our 8.5% error as being able to typically predict a student's letter grade. The model will often times predict if a student will fail the exam, but can only serve as a rough guide. We recommend that advisors seek out students who are predicted to score below the average. This model, used in conjunction with other sources of student performance prediction, will serve its purpose as an early alert system.

## References
- [ ] Cortez, Paulo. (2014). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.



