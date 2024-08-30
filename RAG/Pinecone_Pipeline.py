from datasets import load_dataset
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from pinecone import Pinecone
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter

class DataIngestionPipeline:
    """
    A class to manage the ingestion of data into a vector store.

    This class handles loading data from a dataset, preparing it for ingestion, setting up the Pinecone vector store,
    configuring an ingestion pipeline, and running the ingestion process.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset to be loaded.
    api_key : str
        The API key for Pinecone.
    index_name : str
        The name of the Pinecone index to be used.
    df : pd.DataFrame, optional
        The DataFrame containing the dataset after loading and preprocessing.
    documents : list of Document
        The list of `Document` objects created from the DataFrame.
    pinecone : Pinecone, optional
        The Pinecone client initialized with the API key.
    vector_store : PineconeVectorStore, optional
        The vector store used to manage vectors in Pinecone.
    pipeline : IngestionPipeline, optional
        The ingestion pipeline configured with transformations and storage.
    """

    def __init__(self, dataset_name, api_key, index_name):
        """
        Initialize the DataIngestionPipeline with the given dataset name, Pinecone API key, and index name.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to be loaded.
        api_key : str
            The API key for accessing Pinecone.
        index_name : str
            The name of the Pinecone index to use.
        """
        self.dataset_name = dataset_name
        self.api_key = api_key
        self.index_name = index_name
        self.df = None
        self.documents = []
        self.pinecone = None
        self.vector_store = None
        self.pipeline = None

    def load_data(self):
        """
        Load data from the specified dataset and preprocess it.

        This method loads the dataset using `load_dataset`, converts it to a Pandas DataFrame, and removes the 'hash' column.
        """
        data = load_dataset(self.dataset_name)
        self.df = pd.DataFrame.from_dict(data['fr']).drop(columns=["hash"])
    
    def prepare_documents(self):
        """
        Convert each row of the DataFrame into a `Document` object.

        This method iterates over the rows of the DataFrame and creates a `Document` for each row, which is then added
        to the `documents` list.
        """
        for index, row in self.df.iterrows():
            doc = Document(
                text=row["document"],
                metadata={
                    "id": row["id"],
                    "book": row["book"]
                },
            )
            self.documents.append(doc)
    
    def setup_pinecone(self):
        """
        Initialize the Pinecone client and set up the vector store.

        This method creates a Pinecone client using the provided API key, initializes the Pinecone index, and sets up
        the vector store using the Pinecone index.
        """
        self.pinecone = Pinecone(api_key=self.api_key)
        index = self.pinecone.Index(self.index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=index)
    
    def setup_pipeline(self):
        """
        Configure the ingestion pipeline.

        This method sets up the `IngestionPipeline` with specified transformations and a vector store. The transformations
        include sentence splitting and embedding using a Hugging Face model.
        """
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=150, chunk_overlap=0),
                HuggingFaceEmbedding(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"),
            ],
            vector_store=self.vector_store,
            docstore=SimpleDocumentStore()
        )

    def run(self):
        """
        Execute the entire data ingestion process.

        This method performs all the steps required for data ingestion, including loading the data, preparing documents,
        setting up Pinecone, configuring the pipeline, and running the ingestion.
        """
        self.load_data()
        self.prepare_documents()
        self.setup_pinecone()
        self.setup_pipeline()
        self.pipeline.run(documents=self.documents, show_progress=True)

# Usage example
if __name__ == "__main__":
    dataset_name = "HFforLegal/laws"
    api_key = "PINECONE_API_KEY"
    index_name = "eunomia"

    ingestion_pipeline = DataIngestionPipeline(dataset_name, api_key, index_name)
    ingestion_pipeline.run()
