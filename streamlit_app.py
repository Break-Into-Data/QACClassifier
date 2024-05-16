import requests

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app setup
st.set_page_config(page_title="Text Classification App", page_icon="🤖")

# Streamlit app setup
st.title('Text Classification App')
st.markdown("""
    ### Predict if the given text is a Question, Answer, or Comment
    Enter some text below and click "Classify" to see the predicted probabilities for each category.
""")

# Input text from user
input_text = st.text_area('Enter text to classify:')

if st.button('Classify'):
    if input_text:
        # Send request to the server
        response = requests.post('http://localhost:3000/classify', json={'text': input_text})
        probabilities = response.json().get('result', [])

        if probabilities:
            # Convert probabilities to percentages
            percentages = [p * 100 for p in probabilities]

            # Display probabilities
            col1, col2, col3 = st.columns(3)
            col1.metric('Question', f'{percentages[0]:.2f}%')
            col2.metric('Answer', f'{percentages[1]:.2f}%')
            col3.metric('Comment', f'{percentages[2]:.2f}%')

            # Display probabilities in a bar chart
            categories = ['Question', 'Answer', 'Comment']
            prob_dict = {categories[i]: percentages[i] for i in range(len(categories))}
            
            st.markdown("### Classification Probabilities")
            fig, ax = plt.subplots()
            sns.barplot(x=list(prob_dict.keys()), y=list(prob_dict.values()), palette='viridis', ax=ax)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Category")
            ax.set_ylabel("Probability (%)")
            for index, value in enumerate(percentages):
                ax.text(index, value + 1, f"{value:.2f}%", ha='center')
            st.pyplot(fig)
        else:
            st.error('Failed to get a response from the server.')
    else:
        st.write('Please enter some text.')
