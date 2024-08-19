import nest_asyncio
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv, find_dotenv

from createPDFfromMDClass import MarkdownToPDFConverter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient
from io import StringIO

from langchain.agents import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus.vectorstores import Milvus

class ParseMDClass:
    def __init__(self, pdf_path, output_path,  db_name, collection_name, uri_connection=None, host=None, port=None, llama_key=None, openai_key=None):
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.db_name = db_name
        self.collection_name = collection_name
        self.uri_connection = uri_connection or "http://localhost:19530"
        self.host = host or "localhost"
        self.port = port or 19530
        
        self.llama_key = llama_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        self.openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        
        # Permite que asyncio se ejecute en un entorno no asincrónico
        nest_asyncio.apply()
        
        # Cargar las variables de entorno
        load_dotenv(find_dotenv(), override=True)
    
    def process_pdf(self):
        # Crear instancia de LlamaParse
        parser = LlamaParse(
            api_key=self.llama_key,
            result_type="markdown",  
            parsing_instruction="""
                - Elimina encabezados y pies de pagina de cada pagina del PDF.
                - Organiza tablas en lineas de texto.
                - organiza todo el texto en una sola pagina separada por parrafos.
                - Los párrafos no finalizados complétalo con la información existente en la página siguiente, o en el párrafo siguiente.
                - Organiza en párrafos separados por dos retornos de carro.
                - Traduce el texto al español.
            """,
            language="es",
            skip_diagonal_text=False,
            invalidate_cache=False,
            gpt4o_mode=True,
            gpt4o_api_key=self.openai_key,
            verbose=True,
            show_progress=True,
        )

        # Cargar y procesar el PDF
        documents = parser.load_data(self.pdf_path)
        txt = "".join(doc.text for doc in documents)
        
        pdf = MarkdownToPDFConverter(txt, "output.pdf")
        pdf.convertir_a_pdf()
        
        converter = MarkdownToPDFConverter(txt, f'{self.db_name}_{self.collection_name}_output.pdf')
        converter.convertir_a_pdf()
        
        headers_to_split_on = [
            
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        # Dividir en encabezados
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(txt)

        # División recursiva de caracteres
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_overlap=0)
        splits = text_splitter.split_documents(md_header_splits)

        return splits

    def connect_milvus(self):
        client = MilvusClient(uri=self.uri_connection, token="joaquin:chamorro")
        connections.connect(alias="default", host=self.host, port=self.port)

        if self.db_name not in db.list_database():
            db.create_database(self.db_name, using="default")
            db.using_database(self.db_name, using="default")
        else:
            db.using_database(self.db_name, using="default")

        print(f"Conectado a la base de datos {self.db_name}")

        return client

    def create_collection(self, client, docs_output):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.openai_key)

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64}
        }

        if self.collection_name not in client.list_collections():
            vector_store = Milvus.from_documents(
                docs_output,
                embedding=embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.uri_connection},
                primary_field="pk",
                text_field="text",
                vector_field="vector",
                index_params=index_params,
                enable_dynamic_field=True,
                drop_old=True
            )
            print("Colección creada")
        else:
            vector_store = Milvus(
                embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.uri_connection},
                enable_dynamic_field=True,
                primary_field="pk",
                text_field="text",
                vector_field="vector",
                index_params=index_params
            )
            print("Colección ya existe")

        print(f"Conexión a Milvus-VectorStore establecida.\nConectado a colleccion: {self.collection_name}\n")

    def run(self):
        # Procesar el PDF
        docs_output = self.process_pdf()

        # Conectar a Milvus
        client = self.connect_milvus()

        # Crear la colección en Milvus
        self.create_collection(client, docs_output)

"""
# Ejemplo de uso de la clase
processor = PDFProcessor(
    pdf_path="./EPID.pdf",
    output_path="./EPID.md",
    uri_connection="http://localhost:19530",
    host="localhost",
    port=19530,
    db_name="EPID_MD",
    collection_name="EPID_MD_collection8"
)

processor.run()
"""