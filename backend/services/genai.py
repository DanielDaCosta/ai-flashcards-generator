from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging 
from vertexai.generative_models import GenerativeModel
from langchain_google_vertexai import VertexAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import json


# Configure log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiProcessor:
    def __init__(self, model_name, project) -> None:
        self.model = VertexAI(model_name=model_name, project=project)

    def generate_document_summary(self, documents: list, **args):

        chain_type = "map_reduce" if len(documents) > 10 else "stuff"

        chain = load_summarize_chain(
            llm=self.model,
            chain_type=chain_type,
            **args
        )

        return chain.run(documents) 
    
    def count_total_tokens(self, docs: list):
        temp_model = GenerativeModel("gemini-1.0-pro")

        total = 0
        logger.info("Counting total billable characters...")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters 

        return total

    
    def get_model(self):
        return self.model

class YoutubeProcessor:
    # Retrieve the full transcript

    def __init__(self, genai_processor: GeminiProcessor) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        self.GeminiProcessor = genai_processor
    
    def retrieve_youtube_documents(self, video_url: str, verbose = False):
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)
        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)

        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            logger.info(f"{author} \n {length}\n{title}\n{total_size}\n{total_billable_characters}")

        return result
    
    def format_processed_concepts(self, processed_concepts):
        combined_dict = {}
        
        for d in processed_concepts:
           combined_dict.update(d)

        # Convert combined dictionary into the required format
        formatted_list = [{"term": key, "definition": value} for key, value in combined_dict.items()]
        return formatted_list
    
    def find_key_concepts(self, documents: list, sample_size: int=0, verbose=False):
        # iterate through all document of group size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")
        
        # Optimize sample size no input
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose:
                logging.info(f"No sample size specified. Setting number of documents per sample as 5. Sample size: {sample_size}")
        
        # Find number of documents in each group
        num_docs_per_group = len(documents) // sample_size + (len(documents)% sample_size > 0)

        # Check thresholds of documents in each group
        if num_docs_per_group > 10:
            raise ValueError("Each group has more than 10 documents and output quality will be degraded"
            "significantly. Increase the sample_size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more than 5 documents and output quality is likely to be degraded."
                         "Consider increasing the sample size")

        # Split the documents in chinks of size num_docs_per_group
        groups = [documents[i:i+num_docs_per_group] for i in range(0, len(documents), num_docs_per_group)]

        batch_concepts = []
        batch_cost = 0
        logger.info("Finding key concepts...")
        for group in tqdm(groups):
        
            group_content = ""
            for doc in group:
               group_content += doc.page_content 

            if not group_content:
               logger.warning("No content to process for this group.")
               continue  # Skip to the next group if there's nothing to process

            prompt = PromptTemplate(template="""
                     Find the key concepts and their definitions from the following text:
                     {text}.
                     Respond only in clean JSON format without any labels or additional text. The output exactly should look like this:
                     {{"concept1": "definition1", "concept2": "definition2"}}
                     """, input_variables=["text"])

            chain = prompt | self.GeminiProcessor.model
     
            try:
                output_concept = chain.invoke({"text": group_content})
                
                output_concept = output_concept.replace("```json", "").replace("```", "").strip()

                batch_concepts.append(output_concept)
            except Exception as e:
                logger.error(f"Failed to find concepts for group: {e}")
            
            logger.info(batch_concepts)
            processed_concepts = [json.loads(concept) for concept in batch_concepts]
            
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125
                logging.info(f"Running chain on{len(group)} documents")
                logging.info(f"Total input characters: {total_input_char} ")
                logging.info(f"Total cost: {total_input_cost} ")   
                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000125
                logging.info(f"Total output characters: {total_output_char} ")
                logging.info(f"Total output cost: {total_output_cost} ")   
        
            
        return self.format_processed_concepts(processed_concepts) 
