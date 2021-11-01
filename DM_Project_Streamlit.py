#General libraries
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from streamlit_folium import folium_static
import folium

#Visualisation libraries
import altair as alt
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title('Data Mining Project Question 1 Findings')
st.caption('By: 1181100438 (Azib Jazman) and 1171103938 (Nurina Izzati)')
st.header("Question 1 - Profiling Customer in a Self-Service Coin Laundry Shop")
df = pd.read_csv('CleanData.csv')
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    c1 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Washer_No:O", title="Washer Number"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Washer_No","count()"],
    color=alt.condition(
        alt.datum.Washer_No == 3, 
        alt.value('orange'),
        alt.value('steelblue')
        )
    ).interactive().properties(title = "Number of times each washer is used",width=500,height=300)
    st.altair_chart(c1)

with col2:
    c2 =alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Dryer_No:O", title="Dryer Number"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Dryer_No", "count()"],
    color=alt.condition(
        alt.datum.Dryer_No == 7, 
        alt.value('orange'),
        alt.value('steelblue')
        )
    ).interactive().properties(title = "Number of times each dryer is used",width=500,height=300)
    st.altair_chart(c2)

st.write(
    """
    After appropriately pre-processing the data, we began to explore the data by observing the relationship between each variable. 
    To begin, we look at how many times each washer and dryer are used by the customers.
    From the bar plots above, it seems that washer 3 and dryer 7 are the most used machines by the customers throughout the data collection date.
    """
)

col1, col2, col3= st.columns(3)

with col1:
    c4 =alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Gender:O", title="Gender"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Gender","count()"]
    ).interactive().properties(title = "Number of customers based on Gender",width=400,height=300)
    st.altair_chart(c4)

with col2:
    c5 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Age_Group:O", title="Age Group"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Age_Group","count()"],
    color=alt.condition(
        alt.datum.Age_Group == '40-49', 
        alt.value('orange'),
        alt.value('steelblue')
        )
    ).interactive().properties(title = "Number of customers based on Age Group",width=400,height=300)
    st.altair_chart(c5)

with col3:
    c6 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Body_Size:O", title="Body Size"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Body_Size","count()"],
    color=alt.condition(
        alt.datum.Body_Size == 'fat', 
        alt.value('orange'),
        alt.value('steelblue')
        )
    ).interactive().properties(title = "Number of customers based on Body Size",width=400,height=300)
    st.altair_chart(c6)

col1, col2, col3, col4= st.columns(4)
with col1:
    c7 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("shirt_type:N", title="Sleeve Length"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["shirt_type","count()"],
    ).interactive().properties(title = "Number of customers based on Sleeve Length",width=300,height=300)
    st.altair_chart(c7)

with col2:
    c8 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Shirt_Colour:N", title="Colour Group"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Shirt_Colour","count()"],
    color=alt.condition(
        alt.datum.Shirt_Colour == 'BW', 
        alt.value('orange'),
        alt.value('steelblue')
        )
    ).interactive().properties(title = "Number of customers based on Shirt Colour Group",width=300,height=300)
    st.altair_chart(c8)

with col3:
    c9 =alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("pants_type:N", title="Pants Length"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["pants_type","count()"]
    ).interactive().properties(title = "Number of customers based on Pants Length",width=300,height=300)
    st.altair_chart(c9)

with col4:
    c10 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Pants_Colour:N", title="Colour Group"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Pants_Colour","count()"],
    color=alt.condition(
        alt.datum.Pants_Colour == 'BW', 
        alt.value('orange'),
        alt.value('steelblue')
        )
    ).interactive().properties(title = "Number of customers based on Pants Colour Group",width=300,height=300)
    st.altair_chart(c10)

st.write(
    """
    Following that, we look at the overall number of customers depending on their demographics, such as gender and age group. 
    We discovered the highest category of customers that went to the laundry shop for each demographic, is female gender, and 40-49 age group. 
    Moreover, the majority of the customers who come to the laundry shop wear a short-sleeved shirt with long-length pants of black and white colours for both shirts and pants. 
    We then examined the kind of wash item that customers brought to the laundry shop. 
    Figure below shows that the majority of customers brought a large basket size to the laundry shop, with clothes being the most common type of wash item, presuming that most customers came to the laundry shop to wash their everyday clothes.
    """
)

c11 = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Basket_Size:N", title="Basket Size"), 
    y = alt.Y("count():Q", title="Frequency"),
    color = "Wash_Item:N",
    tooltip = ["Basket_Size","Wash_Item","count()"],
    ).interactive().properties(title = "Type of Wash Item with Basket Size",width=500,height=400)
st.altair_chart(c11)

col1, col2= st.columns(2)
with col1:
    fig = plt.figure(figsize=(5,3))
    sns.countplot(x=df['With_Kids'],order=df['With_Kids'].value_counts(sort="descending").index,hue=df['Age_Group'])
    st.plotly_chart(fig)

with col2:
    fig = plt.figure(figsize=(5,3))
    sns.countplot(x=df['Kids_Category'],order=df['Kids_Category'].value_counts().index,hue=df['Age_Group'])
    st.plotly_chart(fig)

st.write(
    """
    Based on the above plot, we can see that the majority of customers that went to the laundry shop did not bring their children, with customers aged 40-49 having the largest number of no-kids customers. 
    This may indicate that customers who do not have children may have a different size of basket and number of wash items carried into the laundry shop.
    """
)

col1, col2 = st.columns([2,1])

with col1:
    total_cust = alt.Chart(df).mark_bar(size=10).encode(
        x = alt.X("DateTime:T", timeUnit = "monthdate", title="Year"), 
        y = alt.Y("Total Customers:Q", title="Frequency"),
        tooltip = ["DateTime","Total Customers"],
        ).interactive().properties(title = "Number of Customers by Daily",width=800,height=400)
    st.altair_chart(total_cust)

with col2:
    st.header("")
    st.header("")
    st.write(
        """
        From the distribution plot, during the mid October to early of November, the total number of customers coming to the laundry shop was unstable. 
        Sometimes the business might receive up to 60 customers in a day while there will be days when the total customers of the whole day is < 10 customers.
        Moreover, there was a sharp increase number of customers at the end of November between 80-130 total customers per day.
        """
    )
#Question 1
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header('Is there a relationship between basket size and race?')
basket_race = df[["Basket_Size","Race"]]
col1, col2 = st.columns(2)

with col1:
    basket = alt.Chart(basket_race).mark_bar(size=40).encode(
    x = alt.X("Basket_Size:O", title="Basket Size"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Basket_Size","count()"],
    color=alt.condition(
        alt.datum.Basket_Size == 'big', 
        alt.value('orange'),  
        alt.value('steelblue') 
    )
    ).interactive().properties(title = "Types of Basket Size frequently brought buy the customers",width=600,height=400)
    st.altair_chart(basket)

with col2:
    race = alt.Chart(basket_race).mark_bar(size=40).encode(
    x = alt.X("Race:O", title="Race"), 
    y = alt.Y("count():Q", title="Frequency"),
    tooltip = ["Race","count()"],
    color=alt.condition(
        alt.datum.Race == 'indian', 
        alt.value('orange'),  
        alt.value('steelblue') 
    )
    ).interactive().properties(title = "Number of customers based on Race",width=600,height=400)
    st.altair_chart(race)

with col2:
    st.header("")
    st.header("")
    st.subheader("From the correlation, we were able to observe that both basket size and race does not have a strong relationship. We can only see that chinese customers have a extremely low correlation of 0.08. We try to use chi-square to find relationship between basket size and race to support our statement.")

with col1:
    dum_basket_race = pd.get_dummies(basket_race)
    fig = plt.figure()
    sns.heatmap(dum_basket_race.corr(), annot=True, fmt=" .2f", cmap=sns.color_palette("Reds"))
    st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    freq_table = pd.crosstab(basket_race.Race, basket_race.Basket_Size)
    st.table(freq_table)

with col2:
    st.write("Indian customers have a higher probability of bringing a big basket (30.70%) as compared to foreign customers (10.86%). This was also the situation when it came to small basket size, with Indians having a far larger percentage of people bringing small baskets to the laundry business (37.70%) than foreigners (11.48%).")
    st.write("")
    st.write(
        """
        Chinese customers seems to have a slightly higher probability in bringing big basket to the laundry shop (29.76%) 
        as compared with Malay customers(28.68%). In terms of small basket size, Malay customers has way higher chance (34.42%) 
        of bringing it to the laundry shop as compared to Chinese customers (16.39%).
        """
    )

st.subheader("To test whether the insights is statistically significant or not, we conducted the chi-square test of independence.")
st.write(
    """
    - Null Hypothesis: Basket Size and Race are independent of each other.
    - Alternative Hypothesis: Basket Size and Race dependent of each other.
    """)

st.write(
    """
    The Chi-Square test gives us a high p-value = 0.1683. Since the p-value < 0.05, we are unable to
    reject the null hypothesis. Therefore, we can conclude that the basket size and customer's race has no relationship with one another.
    """
    )


#Question 2
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header('When is a good time for the owner to perform washer or dryer maintenance?')

col1, col2 = st.columns(2)

with col1:
    day = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Day:O", title="Day"), 
    y = alt.Y("count():Q", title="No. Of Customers"),
    tooltip = ["Day","count()"],
    color=alt.condition(
        alt.datum.Day == 'Sunday', 
        alt.value('orange'),  
        alt.value('steelblue') 
    )
    ).interactive().properties(title = "Day with the highest customers",width=600,height=400)
    st.altair_chart(day)

with col2:
    time = alt.Chart(df).mark_bar(size=40).encode(
    x = alt.X("Times_Of_Day:O", title="Time Of Day"), 
    y = alt.Y("count():Q", title="No. Of Customers"),
    tooltip = ["Times_Of_Day","count()"],
    color=alt.condition(
        alt.datum.Times_Of_Day == 'Night', 
        alt.value('orange'),  
        alt.value('steelblue') 
    )
    ).interactive().properties(title = "Time of day with the highest customers",width=600,height=400)
    st.altair_chart(time)

st.subheader('As we can see, most customers went to the laundry shop on a weekend (Saturday & Sunday). Now, we wanted to see at which point in time during weekends there are a lot of customers.')

col1, col2 = st.columns(2)
st.write("")
with col1:
    weekend = df[(df['Day']=='Sunday') | (df['Day']=='Saturday')]
    weekend_time = alt.Chart(weekend).mark_bar(size=40).encode(
        x = alt.X("Times_Of_Day:O", title="Time Of Day"), 
        y = alt.Y("count():Q", title="No. Of Customers"),
        tooltip = ["Times_Of_Day","count()"],
        color=alt.condition(
        alt.datum.Times_Of_Day == 'Dawn', 
        alt.value('orange'),  
        alt.value('steelblue') 
        )
        ).interactive().properties(title = "Time of the day with the highest customers during Weekend",width=600,height=300)
    st.altair_chart(weekend_time)

with col2:
    st.write("")
    st.write("")
    st.write("Interestingly, majority of the customers tend to go to the laundry shop at dawn (4am to 8am) on the weekends. We believe that because customers are more focused on their work or studies during the weekday and they would like to enjoy their weekend with leisure, is the reason why most people prefer to do their laundry on dawn time during the weekend.")

col1, col2 = st.columns(2)
weekday = df[~(df['Day']=='Sunday') | (df['Day']=='Saturday')]

with col1:
    weekday_time = alt.Chart(weekday).mark_bar(size=40).encode(
        x = alt.X("Times_Of_Day:O", title="Time Of Day"), 
        y = alt.Y("count():Q", title="No. Of Customers"),
        tooltip = ["Times_Of_Day","count()"],
        color=alt.condition(
        alt.datum.Times_Of_Day == 'Night', 
        alt.value('orange'),  
        alt.value('steelblue') 
        )
        ).interactive().properties(title = "Time of the day with the highest customers during Weekday",width=600,height=300)
    st.altair_chart(weekday_time)

with col2:
    st.write("")
    st.write("")
    st.write("During the weekdays, we found out that there are a lot of customers preferred to do their laundry during the night time (8pm-11:59pm). It could be that during weekdays, people have work or classes to attend, thus the night time is the only available time they have to do their laundry.")


col1, col2, col3 = st.columns([1,2,1])
thursday = df[df['Day']=='Thursday']
with col1:
    st.write("")

with col2:
    thursday_time = alt.Chart(thursday).mark_bar(size=40).encode(
        x = alt.X("Times_Of_Day:O", title="Time Of Day"), 
        y = alt.Y("count():Q", title="No. Of Customers"),
        tooltip = ["Times_Of_Day","count()"],
        color=alt.condition(
        alt.datum.Times_Of_Day == 'Night', 
        alt.value('orange'),  
        alt.value('steelblue') 
        )
        ).interactive().properties(title = "Time of the day with the lowest customers during Thursday",width=700,height=400)
    st.altair_chart(thursday_time)

with col3:
    st.write("")

st.subheader("We found out that on Thursday, the laundry shop will tend to have lower number of customers. Therefore, we would suggest for the owner to perform maintenance to their washing machine/dryer on Thursday Dawn because there will be way less customers affected due to the unavailable washing machine or dryer.")
st.subheader("Moreover, we would like to highly suggest that the owner should made all of his/her washing machines or dryers are in good condition during Saturday and Sunday because there are a lot of customers would be coming to the shop especially around 4am to 11:59am. This is to ensure that customers are able to use the services well and thus increase customer satisfaction.")

#Question 3
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header('Does weather conditions and holidays affect the number of customers to the laundry shop?')
df_q2 = df.copy()
df_q2["DateTime"] = pd.to_datetime(df_q2["DateTime"])
col1, col2= st.columns(2)

with col1:
    st.subheader('Total Customers per day based on Rain Condition')
    fig = plt.figure()
    sns.scatterplot(data=df_q2, x=df_q2.DateTime.dt.date, y='Total Customers', hue='Rain_Condition')
    st.plotly_chart(fig)


with col2:
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.write("Based from the above scatter plot of Rain Condition, we got to observe that most customers tend to come to the laundry shop during day with no rain, assuming that it is a sunny day. However, we also can observe that even during day with having a light rain, customers still come to the laundry shop. This shows that weather might does not have a big impact on the number of customers coming to the laundry shop.")
    st.write("🔵 - No Rain, 🟠 - Light Rain ")

col1, col2 = st.columns(2)

with col1:
    st.subheader('Total Customers per day based on School Holiday')
    fig = plt.figure()
    sns.scatterplot(data=df_q2, x=df_q2.DateTime.dt.date, y='Total Customers', hue='School_Holiday')
    st.plotly_chart(fig)


with col2:
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.write(
        """
        During school holidays, the highest total number of customers in a day was around 140 while it was only 60 total customers when it is not a school holiday season. Furthermore, we found out that on average, there will be approximately 94 customers coming to the laundry shop during the school holiday while there will be only 39 customers during normal days. This shows that the school holiday has an effect on the number of customers to the laundry shop.
        """
    )
    st.write("🔵 - No, 🟠 - Yes ")
    

#Drop columns
df2_q2 = df_q2[["Rain_Condition","Public_Holiday","School_Holiday","Total Customers"]]
dum_q2 = pd.get_dummies(df2_q2)

col1, col2 = st.columns(2)

with col1:
    fig = plt.figure()
    corr_q2 = dum_q2.iloc[:, :].corr()
    sns.heatmap(corr_q2, annot=True, fmt=" .2f", cmap=sns.color_palette("Reds"))
    st.pyplot(fig)

with col2:
    st.write("")
    st.subheader("In terms of weather: ")
    st.write("From the heatmap correlation, it seems that there is a slightly positive correlation between light rain and number of customers with 0.18. This indicates that when the day is having a light rain, more people came to the laundry shop.")
    st.write("For no rain condition, it shows a slightly negative correlation of -0.18 with number of customers indicates that when there is no rain, there is few people at laundry shop.")
    st.write("")
    st.subheader("In terms of School Holidays: ")
    st.write("When there is a school holiday on the day, there is a high positive correlation of 0.62 with number of customers. This means that when it is a school holiday, laundry shop will be having a lot of customers. However, this seems opposite for no school holiday day as it have a high negative correlation of -0.62 with the number of customers.")

#Question 4
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header("Which combination of washers and dryers are used the most by the customers?")
st.write("We decided to identify which combinations of washing machines and dryers are used the most by the customers. This can be done by using the Association Rule Mining technique. In our case, we implemented the Apriori algorithm from the apyori package and the mlxtend package. For both of the packages, we set the minimum support = 5% and minimum confidence = 30%")

st.subheader("Apriori from Apyori package")
apr_apy = pd.read_csv('apyori_results.csv')
apr_apy
st.write("Usually, when we go to the laundry shop, we would wash our clothes and then use the dryers to dry our clothes so that we are left with fresh clean and dry clothes. No customers actually dries their clothes first and then washes their clothes. Therefore, Rule 2 and Rule 4 does not apply in this situation.")
st.write("For Rule 1, we found out that Washer 6 and Dryer 10 were commonly used together. The confidence level of 0.3436 indicates that out of all the customers that uses Washer 6, 34.36% of them would likely use Dryer 10 to dry their clothes. This could be that Dryer 10 is the most nearest to Washer 6 rather than the other washers. This makes sense as people tend to minimize movement and preferred the nearest available dryer to them especially if they bring a bigger load of items to be washed.")

st.subheader("Apriori from mlxtend package")
apr_mlx = pd.read_csv('rules_ap.csv')
apr_mlx
st.write("When using the Apriori algorithm from the mlxtend package, we found out that there a lot more rules that the Apriori algorithm from apyori package was unable to identify.")
st.write("For example, we found out that Washer 3 and Dryer 7 were commonly used together. Among all the customers that uses Washer 3 to wash their clothes, 39.47% of those customers would prefer to use Dryer 7 to dry their clothes. This was not highlighted by the Apriori algorithm from the previous package. With a lift of 1.37 shows that both Washer 3 and Dryer 7 are more likely to be used together by a customer. This could also be due to Dryer 7 have the nearest distance from Washer 3. With less distance, less movement are needed, thus more preferred by the customers.")
st.subheader("To further support our results, we decided to look for the correlation between all the washers and dryers.")

q3 = df[['Washer_No','Dryer_No']]
dum_q3 = pd.get_dummies(q3, columns = ['Washer_No','Dryer_No'], prefix = ['Washer','Dryer'])

col1, col2 = st.columns([1,2])

with col1:
    fig = plt.figure()
    sns.heatmap(dum_q3.corr(), annot=True, fmt=" .2f", cmap=sns.color_palette("Blues"))
    st.pyplot(fig)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("To no suprise, Washer 3 has a weak positive correlation with Dryer 7 (correlation = 0.15) which is inline with the results in our Apriori algorithm from the mlXtend package. This indicates that, if the number of usage for Washer 3 increases, usage of Dryer 7 will also likely increases and vice versa. This also applies to Washer 4 and Dryer 8 (0.11), Washer 5 and Dryer 9(0.10), as well as Washer 6 and Dryer 10 (0.11). Therefore, further strengthen our results.")

#Question 5
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header("What type of customers will likely choose Washer No. 3?")
st.write("From our EDA, we found out that Washer 3 is the most used washer among the four washers. Therefore, we were curious on what kind of customers will likely use Washer No. 3 given customers attributes, whether it is school holiday or not and its rain condition.")
st.write("We have conducted multiple experiments and found out that by using the Top 30 features with the highest scores from Feature Selection using Select K-Best Chi-Squared Test.")
st.write("The top 30 features for this problem are as followed:")
w3_features = pd.read_csv('washer3_features.csv')

col1, col2, col3 = st.columns(3)
col2.dataframe(w3_features)

st.write("We then apply SMOTE to our dataset. This is because originally, our target variable is very imbalanced with a 228 vs 579 for customer uses Washer 3 or not. Therefore, we applied SMOTE in order to treat this imbalance dataset. Our dataset was then balanced to a 410 for both not using (0) and use(1) Washer 3.")
col1, col2 = st.columns(2)

clf_W3 = pd.read_csv('clfW3Metrics.csv')

with col1:
    x=clf_W3['Model_Name']
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=x, y=clf_W3['Precision'],text=clf_W3['Precision'],textposition='auto',),
        go.Bar(name='Recall', x=x, y=clf_W3['Recall'],text=clf_W3['Recall'],textposition='auto',),
        go.Bar(name='F1_Score', x=x, y=clf_W3['F1_Score'],text=clf_W3['F1_Score'],textposition='auto',),
        go.Bar(name='Accuracy', x=x, y=clf_W3['Accuracy'],text=clf_W3['Accuracy'],textposition='auto',)])
    col1.write(fig)

with col2:
    st.header("")
    st.header("")
    st.write(
        """
        We were able to see from the four classification models that every model was capable of producing excellent performance with a score of more than 70%. 
        When given weather, school holiday conditions and consumer characteristics, we determined that the KNN classifier model performs better at predicting the type of customers who will pick Washer 3 with high accuracy, precision, recall and f1-score with 0.81, 0.83, 0.82 and 0.81 respectively. 
        """
    )
    st.write("")
    st.write(
        """
        Although in terms of precision KNN is the lowest and all the other models have more than 84% of precision score, KNN's precision and accuracy score does not have a big difference as compared with other classifiers.
        Moreover, KNN's metrics score are approximately similar to each other.
        Therefore, we believe that KNN classifier were able to perform the best instead of the other models.
        """
        )

#Experiment results
@st.cache
def get_data():
    path = r'expW3.csv'
    return pd.read_csv(path)

expW3 = get_data()

st.subheader("We have conducted multiple experiments on our classification. Please select your conditions below to display the experiments result.")
model = expW3['Model_Name'].unique()
model_choice = st.selectbox('Select Classification Model:', model)
fs = expW3['Features'].unique()
fs_choice = st.selectbox('Select Number of Features:', fs)
sm = expW3['SMOTE'].unique()
sm_choice = st.selectbox('Apply SMOTE:', sm)

a = expW3['Accuracy'].loc[(expW3['Model_Name']==model_choice) & (expW3['Features']==fs_choice) & (expW3['SMOTE']==sm_choice)]
p = expW3['Precision'].loc[(expW3['Model_Name']==model_choice) & (expW3['Features']==fs_choice) & (expW3['SMOTE']==sm_choice)]
r = expW3['Recall'].loc[(expW3['Model_Name']==model_choice) & (expW3['Features']==fs_choice) & (expW3['SMOTE']==sm_choice)]
f = expW3['F1_Score'].loc[(expW3['Model_Name']==model_choice) & (expW3['Features']==fs_choice) & (expW3['SMOTE']==sm_choice)]
final = pd.concat([p,r,f,a], axis=1)

col1, col2 = st.columns(2)

with col1:
    st.header("")
    st.header("")
    st.write(final)

with col2:
    x = expW3['Model_Name'].loc[(expW3['Model_Name']==model_choice)]
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=x, y=final['Precision'],text=final['Precision'],textposition='auto',),
        go.Bar(name='Recall', x=x, y=final['Recall'],text=final['Recall'],textposition='auto',),
        go.Bar(name='F1_Score', x=x, y=final['F1_Score'],text=final['F1_Score'],textposition='auto',),
        go.Bar(name='Accuracy', x=x, y=final['Accuracy'],text=final['Accuracy'],textposition='auto',)])
    st.write(fig)

st.subheader("Then, we tried to predict what kind of customer will use washer 3 provided the weather and school holiday conditions as well as customer attributes by using the KNN classifier.")
st.write(
    """
    We found out that when it is on normal days with no rain occurring, female Malay customers who are fat and age between 50-59 that did not bring kids to the laundry shop while wearing a secondary-coloured long-sleeved shirt and 
    either pink, grey or brown long pants and brought a big purple, green or orange coloured basket full of clothes will most likely use Washer 3 when they arrive at the shop.
    """)

#Question 6a
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header("How many times would Washer 3 be used in an hour?")
st.write(
    """
    Based from early observation, it shows that Washer 3 are the most popular choice by the customers when they arrived to the laundry shop.
    Hence, we decided to predict the number of usage for Washer 3 in an hour.
    
    For this problem, we decided to create a new dataframe that consists of Customer Demographic such as gender, race, body size, and usage of washer and dryer by daily. The date will be formatted to date with hour.
    We predict the number of usage washer 3 based on customer's demographic(gender, race, age group, body size) and the usage of each washers and dryers by using different regression model such as Linear Regression, Decision Tree Regressor and XGBoost Regressor.
    """
)
st.write("For this regression problem, we decided to use the Boruta method to determine which independent features have a bigger impact to the target variable. Boruta algorithm works well in this case because our dataset contains both categorical and numerical features. From Boruta, we then only take the top 18 features as our threshold. We found out that our models have a lower accuracy error by selecting the top 18 features.")
st.write(" The features for this problem are as followed:")
q6a_features = pd.read_csv('w3Reg_features.csv')

col1, col2, col3 = st.columns(3)
col2.dataframe(q6a_features)

q6a_metrics = pd.read_csv('regW3Metrics.csv')
st.subheader("Results: ")
col1, col2 = st.columns(2)
with col1:
    x=q6a_metrics['Model_Name']
    fig = go.Figure(data=[
        go.Bar(name='R2-Score', x=x, y=q6a_metrics['R2-Score'],text=q6a_metrics['R2-Score'],textposition='auto',),
        go.Bar(name='MAE', x=x, y=q6a_metrics['MAE'],text=q6a_metrics['MAE'],textposition='auto',),
        go.Bar(name='MSE', x=x, y=q6a_metrics['MSE'],text=q6a_metrics['MSE'],textposition='auto',),
        go.Bar(name='RMSE', x=x, y=q6a_metrics['RMSE'],text=q6a_metrics['RMSE'],textposition='auto',)])
    col1.write(fig)

with col2:
    st.header("")
    st.header("")
    st.header("")
    st.write(
        """
        The Multiple Linear Regression models had the lowest error value of 0 for MAE, MSE, and RMSE between the three regression models. 
        The error value is described as how near the predicted values are to the actual values. 
        In our scenario, linear regression was able to precisely match the predicted results of the number of usage of washer 3 in a particular hour based on the customer’s demographic and the total usage of washers and dryers. 
        XGBoost and Decision Tree regression performances are not far from each other with XGBoost performing slightly better. 
        Although all models appear to have low error levels before using feature selection, we believe that using feature selection can significantly improve the regression models performance.
        """
    )
st.subheader("Then, we tried to predict the number of times washer 3 will be used given the number of customers arriving with their attributes as well as the total usage of all washers and dryers in an hour by using Multiple Linear Regression.")
st.write(
    """
    When there are two female customers arrive at the laundry shop in a particular hour, both of whom are Malay and Indian, in the age range of 30-39, with a moderate and fat body size, and both of them use dryer 9, 
    Washer 3 will be used at least once in that hour.
    """)

#Question 6b
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header("How many times would Dryer 7 be used in an hour?")
st.write(
    """
    Based on the ARM results, washer 3 and Dryer 7 are usually used together as a pair by customers.
    Therefore, we decided to see the number of times Dryer 7 would be use in a particular hour.
    """
    )
st.write(
    """
    Furthermore, we followed the identical dataframe and feature selection strategy as in the previous problem with the same
    threshold value of the top 18 features. 
    """
    )
st.write(" The features selected for this question are as followed:")
q6b_features = pd.read_csv('d7Reg_features.csv')

col1, col2, col3 = st.columns(3)
col2.dataframe(q6b_features)

q6b_metrics = pd.read_csv('regD7Metrics.csv')
st.subheader("Results: ")
col1, col2 = st.columns(2)
with col1:
    x=q6b_metrics['Model_Name']
    fig = go.Figure(data=[
        go.Bar(name='R2-Score', x=x, y=q6b_metrics['R2-Score'],text=q6b_metrics['R2-Score'],textposition='auto',),
        go.Bar(name='MAE', x=x, y=q6b_metrics['MAE'],text=q6b_metrics['MAE'],textposition='auto',),
        go.Bar(name='MSE', x=x, y=q6b_metrics['MSE'],text=q6b_metrics['MSE'],textposition='auto',),
        go.Bar(name='RMSE', x=x, y=q6b_metrics['RMSE'],text=q6b_metrics['RMSE'],textposition='auto',)])
    col1.write(fig)

with col2:
    st.header("")
    st.header("")
    st.write("After handling multiple experiments, we found out that two models, Multiple Linear Regression and XGBoost regression, perform the best before applying feature selection while Decision Tree perform better with feature selection of 17 features. We did not conclude the number of suitable features to be used for predicting the usage of dryer 7 as all features show very big influences in predicting the usage of dryer 7 in an hour based on customers’ demographic and usage of other washers and dryers.")
    st.subheader("")
    st.write(
        """
        Out of all three regression models, we found out that Multiple Linear Regression shows the best performance with having a 0.57 for r2-score and very low error of MAE (0.48), MSE (0.43), and RMSE (0.66) 
        in predicting the number of usage for dryer 7 in an hour based on the customer’s demographic and total usage of the machines. 
        """
    )

st.subheader("Then, we tried to predict the number of times dryer 7 will be used given the number of customers in the shop with their attributes as well as the total usage of all washers and dryers in an hour by using Multiple Linear Regression.")
st.write(
    """
    If there were one Indian and one Chinese customers where both of them are female and are between 40-49 years old, as well as one of them has a moderate body size while the other has a large body size, one uses Washer 3 while the other uses Washer 6, 
    Dryer 7 will most likely be used at least once during the same hour.

    """)
#Question 8
st.write('----------------------------------------------------------------------------------------------------------------------------------------------')
st.header("Can the customers be grouped together?")
st.write("In this project, we have been given a lot of information on the customer's characteristics. We were curious on whether if the laundry shop's customers have some similarity between each other and can be segmented together into multiple clusters.")
st.write("Therefore, we implemented K-Modes clustering algorithm to divide the customers into suitable groups.")
st.write("By using the Elbow method, we were able to determine that the optimal number of clusters to be produced is 5 clusters.")

#Clustering file
laundry_customer = pd.read_csv("laundry_customer.csv")

col1, col2, col3, col4 = st.columns(4)

with col1:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Race'],order=laundry_customer['Race'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col2:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Gender'],order=laundry_customer['Gender'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col3:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Age_Group'],order=laundry_customer['Age_Group'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col4:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Body_Size'],order=laundry_customer['Body_Size'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col1:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['With_Kids'],order=laundry_customer['With_Kids'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col2:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Kids_Category'],order=laundry_customer['Kids_Category'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col3:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Basket_Size'],order=laundry_customer['Basket_Size'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col4:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Basket_colour'],order=laundry_customer['Basket_colour'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col1:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Spectacles'],order=laundry_customer['Spectacles'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col2:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Attire'],order=laundry_customer['Attire'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col3:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['shirt_type'],order=laundry_customer['shirt_type'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col4:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['pants_type'],order=laundry_customer['pants_type'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col2:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Shirt_Colour'],order=laundry_customer['Shirt_Colour'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

with col3:
    fig = plt.figure()
    sns.countplot(x=laundry_customer['Pants_Colour'],order=laundry_customer['Pants_Colour'].value_counts().index,hue=laundry_customer['cluster_label'])
    st.pyplot(fig)

st.subheader("Cluster 0")
st.write(
    """
    Cluster label 0 has a larger proportion of male Indian customers aged from 20-29 and 40-49 with fat or thin body size. 
    They will either bring a primary-coloured(red, blue, yellow) big or small basket sizes to the laundry shop. 
    Customers in this cluster will typically not wear a spectacle and wear a short-sleeved shirt, long-length pants in black and white colours.
    """
)

st.subheader("Cluster 1")
st.write(
    """
    Cluster 1 is made up of female Chinese customers between the ages of 30 and 39 who have a moderate body size. 
    They will bring a small child or baby to the laundry shop, along with a large black or white basket. 
    This cluster is more for those who go to do their laundry in casual apparel, such as a short-sleeved, black or white coloured shirt and short-length, red, blue or yellow coloured pants without wearing any spectacles.
    """
)

st.subheader("Cluster 2")
st.write(
    """
    Cluster 2 consists primarily of female Malay clients between the ages of 50 and 59 who have a moderate body size. 
    Customers will arrive at the laundry shop with a large black or white basket. 
    Customers in this cluster will dress in casual attire, with a long-sleeved shirt in primary colours and long-length black or white pants. In addition, this type of customer would not wear glasses.
    """
)

st.subheader("Cluster 3")
st.write(
    """
    Cluster 3 customers are female Malays in their 50s and 60s with a slim build. 
    They were not going to bring any children to the shop. It is very likely that their basket will be a big black or white basket. 
    Customers will likewise dress casually, with a black or white short-sleeved shirt and a pair of pink, grey, or brown long pants and no spectacles.
    """
)

st.subheader("Cluster 4")
st.write(
    """
    Customers in Cluster 4 are Chinese men between the ages of 30 and 39 who have a slim build. 
    They will not bring their kids to the laundry. 
    In terms of the basket they will bring, it is most likely will be a large black or white one. 
    When they go to the store, they will wear a short-sleeved shirt in red, blue, or yellow coloured, along with a pair of short-length black or white pants and no spectacles.
    """
)