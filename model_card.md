# Model Card

## Model Details

For this project, I built a Random Forest model to predict whether someone makes more than $50,000 a year. I used Python and scikit-learn to train the model using census data. The model uses 100 decision trees, and I set the random state to 42 so the results would be consistent if I ran it again.

## Intended Use

The purpose of this model is to show how machine learning can be used to make predictions based on existing data. It looks at information like age, education, occupation, and hours worked to predict whether someone earns more than $50,000 annually.

This model was created for educational purposes and isn't meant to be used for actual financial, hiring, or other important decisions about people.

## Training Data

I used the Census Income dataset from the UCI Machine Learning Repository. It includes information about people's education, jobs, ages, marital status, and other characteristics, along with whether they earn more than $50,000 per year.

I split the dataset so that 80% was used for training and the remaining 20% was saved for testing. Before training, I used one-hot encoding to convert categorical information into numbers that the model could work with. I also converted the income labels into binary values.

## Evaluation Data

I tested the model using the 20% of data that was set aside during the initial split. This gave me a way to see how well the model could make predictions on data it hadn't been trained on.

I also evaluated the model across different categories, including education, occupation, race, sex, and marital status. This helped me look at whether the model performed differently depending on the group being evaluated. The results for each category are saved in `slice_output.txt`.

## Metrics

I used precision, recall, and F1 score to measure how well the model performed.

Precision tells me how often the model was correct when it predicted someone earned more than $50,000. Recall measures how many of the people who actually earned more than $50,000 were correctly identified. F1 score combines precision and recall into one measurement.

After testing the model, I got the following results:

- **Precision:** 0.7353 (73.53%)
- **Recall:** 0.6378 (63.78%)
- **F1 Score:** 0.6831 (68.31%)

Overall, the model did a decent job identifying people in the higher-income category, but there is still room for improvement. The recall score shows that it missed some people who actually earned more than $50,000.

## Ethical Considerations

One concern with this model is that the dataset includes personal characteristics like race and sex. These factors could introduce bias into the predictions, especially if the original data reflects existing inequalities.

Even though I evaluated the model across different categories, that doesn't necessarily mean the predictions are fair for everyone. I think it's important to consider how models like this could affect people if they were used to make real decisions.

Because of these concerns, I wouldn't recommend using this model to make decisions about someone's employment, finances, or opportunities.

## Caveats and Recommendations

One limitation is that the census data is older, so it might not accurately represent people's incomes today. The model also depends on the quality of the data it was trained on.

I only used one train-test split, so testing the model with different splits or cross-validation could help determine how consistent the results are.

If I were going to improve this project, I would try adjusting the model's settings, testing different approaches, and looking more closely at differences in performance between demographic groups.

Overall, I think the model works well for demonstrating the machine learning process, but I would want to do more testing before considering any real-world use.