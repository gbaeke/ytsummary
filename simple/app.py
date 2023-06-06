
import openai
import dotenv
import os
import tiktoken

# get OpenAI key env
dotenv.load_dotenv()

encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# cost per 1000 tokens in France Central
cost_gpt4_32k_input = 0.057
cost_gpt4_32k_outpiut = 0.113

def token_cost(numtokens, priceper1000):
    return numtokens * priceper1000 / 1000


from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import GoogleApiYoutubeLoader

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=-IJspV1HwGk")

transcript = loader.load()

# check the transcript
print(transcript)

from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# use Azure chat model; requires gtp-4 deployment on given endpoint
model = AzureChatOpenAI(
    client=None,
    openai_api_base=os.getenv("ENDPOINT", ""),
    openai_api_key=os.getenv("API_KEY", ""),
    openai_api_version="2023-03-15-preview",
    deployment_name=os.getenv("DEPLOYMENT", ""),
    openai_api_type="azure",
    max_tokens=4096,
    temperature=0,
    verbose=True
)

system_template = "You are a helpful assistant that summarizes Youtube transcripts"
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "Summarize the transcript of this video. If there are different sections, provide more details about each section: {transcript}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

chain = LLMChain(llm=model, prompt=chat_prompt)

result = chain.run(transcript=transcript)

print(result)









