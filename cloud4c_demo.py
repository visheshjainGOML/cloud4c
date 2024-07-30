import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import numpy as np
import re
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key
# Initialize OpenAI client
client = OpenAI()
# Configuration for OpenAI API
openai_config = {
   "temperature": 1,
   "max_tokens": 256,
   "top_p": 1,
   "frequency_penalty": 0,
   "presence_penalty": 0
}
# The prompt to guide the model's response
prompt=""""Given a user query, your role is to analyze and classify it into one of the following categories: Network & Wireless, Accessories, Access Requests, VPN access, Jira Requests, Data Center Services, Data Center Services /Report VM issue, Data Center Services /VM Request, Software Request, Email Services, Hardware, Hardware Issue, Network, Newhire Accounts, Password Reset, Software Installation & Configuration. Additionally, determine the sentiment of the query from the options: Very Positive, Positive, Neutral, Negative, Very Negative. Start by identifying keywords or phrases that suggest a category, then assess the tone to estimate the sentiment. Subsequently, provide a deep analysis of the query with a step-by-step solution to the problem, ensuring your tone of voice is professional. After your analysis, format your response as JSON with the category, sentiment, and solution fields."
You should strictly  only give a json as the output in the specified format.
For example, if the user query is 'I need help setting up my VPN. It's quite urgent as I have a deadline approaching.', your analysis might go like this: The mention of 'VPN' suggests the 'VPN access' category. The urgency and deadline imply a stressed tone, which could be considered 'Negative'. A professional, step-by-step solution could involve verifying network connections, ensuring VPN software is up to date, and contacting IT support if issues persist. Hence, your JSON response should be: {"category": "VPN access", "sentiment": "Negative", "solution": "1. Verify your network connection. 2. Ensure your VPN software is updated. 3. Restart your VPN service. 4. If the problem persists, contact IT support for further assistance."}
"""
 
def extractor(query,prompt =None):
    if prompt == None:
        prompt=""""Given a user query, your role is to analyze and classify it into one of the following categories: Network & Wireless, Accessories, Access Requests, VPN access, Jira Requests, Data Center Services, Data Center Services /Report VM issue, Data Center Services /VM Request, Software Request, Email Services, Hardware, Hardware Issue, Network, Newhire Accounts, Password Reset, Software Installation & Configuration. Additionally, determine the sentiment of the query from the options: Very Positive, Positive, Neutral, Negative, Very Negative. Start by identifying keywords or phrases that suggest a category, then assess the tone to estimate the sentiment. Subsequently, provide a deep analysis of the query with a step-by-step solution to the problem, ensuring your tone of voice is professional. After your analysis, format your response as JSON with the category, sentiment, and solution fields."
You should strictly  only give a json as the output in the specified format.
For example, if the user query is 'I need help setting up my VPN. It's quite urgent as I have a deadline approaching.', your analysis might go like this: The mention of 'VPN' suggests the 'VPN access' category. The urgency and deadline imply a stressed tone, which could be considered 'Negative'. A professional, step-by-step solution could involve verifying network connections, ensuring VPN software is up to date, and contacting IT support if issues persist. Hence, your JSON response should be: {"category": "VPN access", "sentiment": "Negative", "solution": "1. Verify your network connection. 2. Ensure your VPN software is updated. 3. Restart your VPN service. 4. If the problem persists, contact IT support for further assistance."}

"""
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": str(query)}
    ]
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=prompts,
        temperature=openai_config["temperature"],
        max_tokens=openai_config["max_tokens"],
        top_p=openai_config["top_p"],
        frequency_penalty=openai_config["frequency_penalty"],
        presence_penalty=openai_config["presence_penalty"]
    )
    return response.choices[0].message.content

def analyze_and_display_queries(queries):
    results = []
    for query in queries:
        ans = extractor(query)
        print(ans)
        ans_eval = eval(ans)
        results.append([query, ans_eval["category"], ans_eval["sentiment"]])
    df = pd.DataFrame(results, columns=["Description", "Category", "Sentiment"])
    df["Sentiment Rank"] = df["Sentiment"].map({"Very Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Very Positive": 4})
    df.sort_values("Sentiment Rank", inplace=True)
    df.drop("Sentiment Rank", axis=1, inplace=True)
    st.write(df)
 
 
def ticket_category_app():
    st.title("CtrlS Ticketing Analyzer")
    user_input = st.text_area("Enter your query", height=150)
    uploaded_file = st.file_uploader("Add Attachment (CSV Log File)", type=['csv'])
 
    if user_input and uploaded_file:
        data = pd.read_csv(uploaded_file)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
 
        # Simulate 'Packet Loss' and 'Latency' if not present
        if 'Packet Loss' not in data.columns:
            data['Packet Loss'] = np.random.randint(0, 100, size=len(data))
        if 'Latency' not in data.columns:
            data['Latency'] = np.random.randint(50, 500, size=len(data))
 
        st.subheader("Packet Loss Over Time")
        st.line_chart(data.set_index('Timestamp')['Packet Loss'])
 
        st.subheader("Latency Over Time")
        st.line_chart(data.set_index('Timestamp')['Latency'])
 
        # Analyzing the query
        with st.spinner('Analyzing your query...'):
            ans = extractor(user_input)
        ans_eval = eval(ans)

        st.subheader("Identified Category:")
        st.success(ans_eval["category"])
       
        st.subheader("Identified Sentiment:")
        st.success(ans_eval["sentiment"])
 
        st.subheader("Identified Solution:")
        ans_eval_solution = re.sub(r'\d+.', '', ans_eval["solution"])

        solution_items =ans_eval_solution.split('.')
       
        solution_formatted = "\n".join([f"{i+1}. {item.strip()}" for i, item in enumerate(solution_items) if item.strip() != ""])
        st.markdown(solution_formatted)
        with st.spinner('Generating Response to Customer...'):
           
            response = extractor(query=f"You will be give with Given a user query. this is the user query {user_input}. you have to craft a response to the user back as the same language as the user wrote in something sort of this way 'Thank you for reaching out. Your query regarding {ans_eval['category']} is noted. The estimated resolution time is based on current load and complexity. Meanwhile, try the following steps:\n{solution_formatted}. You must respond in the same language the query is been raised for example if the user query is in swedish the output has to be in the same language ", prompt = "You are an helpful assistant, give only the answer not start the output with 'Certainly!' or sentence based with your assitance ")
            st.subheader("Customer Response:")
            # f"Thank you for reaching out. Your query regarding {ans_eval['category']} is noted. The estimated resolution time is based on current load and complexity. Meanwhile, try the following steps:\n{solution_formatted}"
            st.info(response)

        with st.spinner('Finding the possible causes..'):
            response = extractor(user_input,prompt ="You will be give with Given a user query, your role is to analyze and find the route cause of the problem and list it")
            st.subheader("Plausible area of investigation:")
            st.info(response)

        
 
        # Display relevant log entries
        relevant_logs = data[data['Message'].str.contains("outage|packet loss|latency", case=False, regex=True)]
        st.subheader("Relevant Log Entries")
        st.write(relevant_logs)


 
    with st.sidebar:
        st.subheader("Bulk Query Analysis")
        query_count = st.number_input('How many queries would you like to analyze?', min_value=1, value=1, step=1)
        queries = [st.text_area(f"Query-{i+1}", key=f"query-{i}") for i in range(query_count)]
        analyze_button = st.button("Analyze Queries")
        if analyze_button:
            analyze_and_display_queries(queries)
 
if __name__ == "__main__":
    ticket_category_app()