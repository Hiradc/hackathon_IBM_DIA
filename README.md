### I. Creation process

During this hackathon, we began by determining what result we wanted, what we wanted to develop. What was our goal, our outcome? To do this, we spent our Wednesday afternoon brainstorming our concept, determining its outlines, while remaining aware of the technical challenges.

So, we initially started with a web extension that acts as an overlay on the websites of the largest LLMs. When the user connects to one of these LLM sites and interacts with the chatbot, the web extension integrates new data into the website interface with information about each request.

This solution allowed anyone to get started quickly, as all they had to do was download the extension from the web extension stores.

Unfortunately, after numerous tests on Wednesday afternoon, we realised that WatsonX was blocking API requests from the web. This meant that the web extension was unable to query the WatsonX API directly and had to go through an external server, which made the project much more complex and the costs much higher.

So we revised our objective and decided instead to go for a web interface that mimics the major chatbot sites (such as ChatGPT and Claude) and allows users to interact with different chatbots. It works a bit like Mamouth, with a portal and a single interface that allows users to talk to different models. This interface, developed entirely by us, gives us much more technical freedom.

But the goal is not just to copy how Mamouth works, but to add our own value by including the carbon footprint of each query.

### II. Technical difficulties

Initially, we planned to host our AI model in WatsonX, but because deploying a model in WatsonX requires storage space and therefore a paid subscription, we were unable to put the AI in WatsonX. As a result, the AI model for predicting electricity consumption and the web interface are hosted locally.

### III. Data Exploration and Preprocessing
The start of our work was the exploration of data, to understand it better and prepare the machine learning part. We read the research paper (https://www.arxiv.org/pdf/2407.16893), and realized that the model used by the researchers was a simple Random Forest model, using very few variables.

We then started our work on the datasets on WatsonX, merging all of them in one single file to simplify the task. After cleaning the new dataset, we dropped some columns either because the information was not interesting, or it was a repetition of another column. For example, there were multiple columns evaluating the duration of the prompt + response (clock duration, start_time, end_time), we removed them but kept another that we deemed clearer. After encoding the columns where the values weren't numerical, we looked at the correlation between each variable and our target variable : energy_consumption_llm_total.
To help the model sclae its predictions to larger LLMs, we added a feature representing the number of parameters in a LLM:  **model_size**

We were expecting a lot of variables to have an effect on the energy consumption, and were therefore surprised by the results : out of the 63 columns we had kept (from 78 original columns), only 4 native columns had a correlation score superior to 0.1. Those columns were the following : **response_token_length**, **response_duration**, **total_duration** and **adj_count**. Our added column **model_size** had a good correlation of almost 0.4. As **response_duration** and **total_duration** had a very similar correlation to the target variable, we removed the **total_duration and**, to avoid having too little variables to train our model on, we decided to put the correlation treshold to 0.09 to add the two next variables with the highest correlation score : **polysyllabcount** and **long_word_count**.

We dropped all the other columns, and saved the dataset to use it to train our future model.

### IV. Training the model

Using the code provided in the github of the researchers (https://github.com/ejhusom/MELODI), we rapidly tested a few models and upon confirming that RandomForestRegressor was the best model, we toggled its parameters to have the best predictions. We also had to consider the model size since we needed to put the model on github. We chose the following parameters: n_estimators=100, random_state=42, max_depth=15, n_jobs=-1 and ended up with mse: 4.039415791317722e-09, mae: 6.896921413127833e-06 and **r²: 0.9849131513059202**.

### V. Carbon footprint fo each request

Once our AI has predicted the electricity consumption of a query, we plug this consumption into the formula below to obtain the carbon footprint of that query.

$$
\text{Carbon footprint of the request} =
\text{Energy consumed by the request} \times 
\text{Carbon footprint of a kWh} +
\frac{\text{Carbon footprint of model training}}{\text{Average number of request}}
$$

In our formula, we chose to take into account the cost of training the model. This is because the majority of a model's carbon footprint comes from its training. Our calculations would therefore be meaningless if we did not take this parameter into account.

### VI. Developping the app

After deciding to use an app for our idea, we brainstormed a bit to choose the deployement architecture, and choose Flask, a micro web framework written in Python which does not require particular tools or libraries, as some members had already used it and were familiar with its uses.

We designed our webpage with Html, and once the external appearance was satisfactory, we begun to search for ways to implement what we wanted to do.

Our plan was for the prompt to be sent to a LLM of the user's choice via the OpenAI API, analyze the response sent back and collect the information we needed to send to our prediction model. The response is dissected with different functions, and we end up with a small dataframe containing its information. The information is given to our model, predicting the CO2 emission equivalent. Finally, the response to our prompt is shown to the user, and the CO2 emission is saved and added to the total emmision, which is then also shown to the user.

The user can freely change between models, the total emission will still be tracked.



