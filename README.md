# <a name="top"></a>Predicting Github Repository Programming Language based on project README files
![](

by: Vincent Banuelos and J. Vincent Shorter-Ivey

***
[[Project Description/Goals](#project_description_goals)]
[[Initial Questions](#initial_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Pipeline Takeaways](#pipeline)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
- Using both basic stats and advanced stats can I predict a github repositories programming langauge based on the repositories README file.

- This project runs through the entire Data Science Pipeline and culminates with classification modelling techniques based upon Natural Langauge Processing outcomes.

[[Back to top](#top)]


## <a name="initial_questions"></a>Initial Questions:

- What are the most common words in READMEs?
- Does the length of the README vary by programming language?
- Do different programming languages use a different number of unique words?
- Are there any words that uniquely identify a programming language?

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
|langauge|the langauge of the repository|category|
---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |



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

### Exploration Summary

- Created Class to hold words, bigrams, and trigrams for each langauge

- dont forget to grab message lenght as a category

- May be worth it to decide if a word is a basketball word or coding word
  - cna them assign them negative or psoitive values and 

## Modeling takeaways

-  

- 

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion, and Next Steps:

- 

- 

- In conclusion,   
    
[[Back to top](#top)]
