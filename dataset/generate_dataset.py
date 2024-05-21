import csv
import concurrent.futures
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os


load_dotenv()

# Set the LANGCHAIN_TRACING_V2 environment variable to 'true'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Set the LANGCHAIN_PROJECT environment variable to the desired project name
os.environ['LANGCHAIN_PROJECT'] = 'Conversation30DayProject'


class Message(BaseModel):
    """
    A model for representing a single message within a message log.
    Inlcudes meta data like the user_name, message_type, and a message_id which acts not only as a unique identifier but conversation sequence identifier.
    """
    user_name: str = Field(..., description="User name who submitted the message")
    message: str = Field(..., description="Full message that the user sent, most likely a question, answer, or comment.")
    message_type: str = Field(...,description="Category of the type of interest the message provokes. Question, Answer, Comment")
    message_id: int = Field(...,description="7 digit number, first 5 is the conversations unique identifier and the last 2 is the message sequence number")


class Conversation(BaseModel):
    """The full message log to review the message history in sequential order."""
    message_history: List[Message]


def extract_desc_fields(input_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at generating synthetic coversations. Your task is to create a range of synthetic conversations that comes from the "Break Into Data" Discord Server.
                    The discord servers's purpose is to help people get data related skills and roles, including data analytics, data science, machine learning, data engineering and ai engineering.
                    The Server has a wide range of skillsets ranging from beginners who just started their data journey to proficient and experienced members. 
                    
                    The variety of backgrounds, experience, and interests helps to invoke interesting conversations throughout the discord server. The primary channels is a general, job search support, content creation, share your project, and resources.
                    - The channel is small enough that there are only 2-5 users active in a conversation at the same time before another topic get picked up on the channel. 
                    - Each conversation should be focused around a single general subject but can include tangents that it runs down before coming to a conclusion.
                    - While each conversation most likely focuses on question and answer conversations it can also include comments that people share about their expierence or even help to build/rephrase the question.
                    - Conversations can also include spam, while this does occure it is less than 15 percent of the messages.
                    - Conversations typically range between 3-15 messages
                    
                    Example of topics include:
                    Linear/logistical Regression, classification, clustering, neural networks, random forests, resume help, interview preparation questions, hackathons, networking events
                    articles, news, youtube videos, coding cookbooks, cheatsheets
                """,
            ),
            ("human", "{text}"),
        ]
    )
    
    llm = ChatGroq(model_name="llama3-70b-8192")
    # llm = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY") , model_name="claude-3-sonnet-20240229")
    # llm = ChatGoogleGenerativeAI(model_name="models/gemini-1.5-pro-latest") # "models/gemini-1.5-flash-latest" or "models/gemini-1.5-pro-latest"

    extractor = prompt | llm.with_structured_output(
        schema=Conversation,
        method="function_calling",
        include_raw=False,
    )
    
    return extractor.invoke(input_prompt)


def generate_conversation():
    return extract_desc_fields("Generate and save the created conversations.")


def write_conversation_to_csv(conversation, writer):
    for message in conversation.message_history:
        writer.writerow([message.user_name, message.message, message.message_type, message.message_id])


def main(conv_filepath, num_conversations=10):
    file_exists = os.path.exists(conv_filepath)

    with open(conv_filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['user_name', 'message', 'message_type', 'message_id'])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(num_conversations):
                future = executor.submit(generate_conversation)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    conversation = future.result()
                    write_conversation_to_csv(conversation, writer)
                except Exception as e:
                    print(f"Error occurred while generating conversation: {str(e)}")
                    continue

if __name__ == "__main__":
    conv_filepath = './data/conversations.csv'
    main(conv_filepath, 10)
