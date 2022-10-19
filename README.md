# <a name="top"></a>Predicting Github Repository Programming Language based on project README files
![]()

by: Vincent Banuelos and J. Vincent Shorter-Ivey

***
[[Project Description/Goals](#project_description_goals)]
[[Executive Sumary](#exec_summ)]
[[Initial Questions](#initial_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Pipeline Takeaways](#pipeline)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
- Using both basic stats and advanced stats can we predict a github repositories programming langauge based on the repositories README file.

- This project runs through the entire Data Science Pipeline and culminates with classification modelling techniques based upon Natural Langauge Processing outcomes.

- Utilizes the Top 1

[[Back to top](#top)]

## <a name="exec_summ"></a>Executive Summary:
- Analysis of the data showed that Python was by far the most popular language

- Python was the most popular language used.

- Creation of bigrams was a huge help to modeling process.

[[Back to top](#top)]


## <a name="initial_questions"></a>Initial Questions:

- What are the most common words in the READMEs?
  - (Word with # of appearances)
    - Team: 341
    - Game: 293
    - Player: 281
    - Data: 263
    - Season: 144

- Does the length of the README vary by programming language?
  - Yes, of the languages that were found, readme length tends to vary based on the language in the readme, with Java Script on average having the smallest README Lengths.  

- Do different programming languages use a different number of unique words?
  - Yes, each programming language has a different number of Unique words.
    - Python: 1733
    - JavaScript: 219
    - HTML: 83
    - Other: 607

- Are there any words that uniquely identify a programming language?
  - 

[[Back to top](#top)]


## <a name="planning"></a>Planning:

- Create README.md with data dictionary, project goals, and come up with initial hypotheses.
- Acquire data from single repository as test. After succesful test, decide on 100 repositories to analyze.
- Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a functions to automate the process. 
- Store the acquisition and preparation functions in a wrangle.py module function, and prepare data in Final Report Notebook by importing and using the function.
- Clearly define at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train at least 3 different classification models.
- Evaluate models on train and validate datasets.
- Choose the model that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.
- Prepare slideshow and recorded presentation for classmate and faculty consumption
  - Slide deck available *here <insert URL to slide deck>*

[[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  

| Target Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|langauge|the langauge of the repository|object|
---
| Feature | Definition | Data Type |
| ----- | ----- | ----- 
| word_count | count of README words  |float64 
| lemmatized | ----- |float64/object
| language_bigrams | word bigrams popular within each language | float64/object
---

## <a name="reproduce"></a>Reproduction Requirements:

You will need your own env.py file with database credentials then follow the steps below:

  - Download the csv files, wrangle.py, model.py, explore.py, and final_report.ipynb files
  - Run the final_report.ipynb notebook

[[Back to top](#top)]


## <a name="pipeline"></a>Pipeline Conclusions and Takeaways:

###  Wrangling Takeaways
- Performed initial web scrapping to pull in top 100 repositories after searching for `baskteball`
- Performed additional scrapping to separate all text content from README
- Used github language identification for labeling of coding language
- Following the Data Acquisition the following preparation work was done to the acquired data:
    - Eliminated non-English language README files --> backport to acquire function
    - Tokenined, Lemmatized, and Cleaned text portion of files
---
### Nulls/Missing Values
* Simple drop of null values as they most often indicated an empty README
---
### Feature Engineering 
* Engineered `word_count` in order to facilitate analysis around column length and unique word density
* Engineered `language_bigrams` in order to capture most used word duos
---   
### Flattening
* Had to make decisions in order to remove optionality from language column due to sample size 
- Went from around 17 languages down to 7 by creating an `other` category for the less popular langauges
* Decisons here driven mostly by desire to have enough observations for effective analysis 
---

### Exploration Summary

- Created Class to hold words, bigrams, and trigrams for each langauge

- dont forget to grab message lenght as a category

- May be worth it to decide if a word is a basketball word or coding word
  - Can them assign them negative or psoitive values and 


# Modeling
- Things did not go as plan. Initially had massive perfomanc drops moving in Validation
- Use of custom class proved to be more of a hindrance than help
- Had to abandon gridsearch idea, and focus on feature creation
- Logistic Regression never provided much performance gain above baseline
- DTC models consistenly peformed well, and, along with RF, we started lowering depth to control for overfitting 
- We did 5 rounds of mass cohort testing before settling on specific hyperparameters
- Final Models had 37% performamce gain above baseline when scoring with Accuracy as focus
--- 

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion, and Next Steps:

# Conclusion
## Summary of Key Findings
* 
* 
* 
* DTC and RF models consistenly performed well
* Final Models had 37% performamce gain above baseline
---
## Suggestions and Next Steps
* Trigrams may be something worth adding in the model in order to boost performance
* We also want to create a form of sentiment analysis 
    - It will track whether a repo leans more towards Basketball or Coding as a focal point
* Model performance above baseline is enough to justify continued use.
* An affirmative next step would be to further expand the scope of testing to capture languages with smaller usage.
[[Back to top](#top)]
