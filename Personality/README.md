# PersonalityProfilerML
PersonalityProfilerML is a machine learning project that uses behavioural and social interaction data to classify individuals as introverts or extroverts. The project explores how everyday activities like time spent alone, frequency of social posts, or reactions to interaction can offer deep insight into personality. Using Python, Scikit-learn pipelines, and visual storytelling, I develop a robust classification model backed by interpretable features.

# Dataset Overview
This dataset includes the following columns. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data)
    
    - Time_spent_Alone: Hours spent alone daily (0–11).
    - Stage_fear: Presence of stage fright (Yes/No).
    - Social_event_attendance: Frequency of social events (0–10).
    - Going_outside: Frequency of going outside (0–7).
    - Drained_after_socializing: Feeling drained after socialising (Yes/No).
    - Friends_circle_size: Number of close friends (0–15).
    - Post_frequency: Social media post frequency (0–10).
    - Personality: Target variable (Extrovert/Introvert).

# Project Workflow
1. **Data Cleaning**
   - removed 402 duplicates
   - encoded categorical variables: Stage_fear - {'Yes': 1, 'No': 0), Drained_after_socializing - {'Yes': 1, 'No': 0}
   - checked for missing values

2. **Exploratory Data Analysis** - Key findings
    - Extroverts slightly outnumber introverts - approximately 1400 to 1200.
    - Extroverts consistently show higher and more varied engagement across both digital and real-world social behaviors, while introverts demonstrate lower, more concentrated patterns. Measurable activity clearly distinguishes the two personality types.
    - Extroverts overwhelmingly report that they do not feel drained after social engagements and show no stage fear, unlike most introverts.
    - Introverts tend to spend more time alone, post less frequently, and have smaller friend circles.

3. **Feature Engineering**
   - encoded categorical (binary) variables: Stage_fear_Yes, Drained_after_socializing_Yes
   - feature selection:
      - X = Time_spent_alone, Stage_fear_Yes, Social_event_attendance, Going_outside, Drained_after_socializing, Friend_circle_size, Post_frequency
      - y = Personality

4. **Modeling**
   - train test split: 80% to 20%
   - Models used:
     - Logistic Regression (base line)
     - Random Forest Classifier
     - Gradient Boosting Classifier

5. **Model Evaluation & Performance**
   - Evaluation metrics: accuracy, confusion matrix, classification report, cross-validation score
   - Model Performance

      | Model              | Test Accuracy | Mean CV Accuracy | Std Dev |
      |--------------------|---------------|------------------|---------|
      | Gradient Boosting  | **0.944**     | **0.922**        | 0.021   |
      | Logistic Regression| 0.918         | 0.917            | 0.028   |
      | Random Forest      | 0.910         | 0.899            | 0.013   |

**Gradient Boosting** was selected as the final model based on the highest test and cross-validated accuracy.

6. **Conclusion**

This project demonstrates that social behavior, both online and offline, can be effectively transformed into meaningful predictors of a person’s personality. By combining exploratory data analysis with visual storytelling and machine learning, I developed a classification pipeline capable of accurately distinguishing between introverts and extroverts. 
The visualizations brought these behavioral patterns to life, revealing features like `Post_frequency` and `Time_spent_Alone` as natural separators. 
Ultimately, the Gradient Boosting model confirmed these insights, identifying `Drained_after_socializing_Yes` as the most influential feature, validating what the data had already made visually clear.
