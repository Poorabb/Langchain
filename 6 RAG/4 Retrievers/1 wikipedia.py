from langchain_community.retrievers import WikipediaRetriever

re = WikipediaRetriever(top_k_results=2,lang="en")

result= re.invoke("indian cricketers")

print(result[0].page_content)
