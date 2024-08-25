import os
import streamlit as st
import redis
import datetime
import ast

from createPDFfromMDClass import MarkdownToPDFConverter

from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient

from dotenv import load_dotenv, find_dotenv
from time import sleep


from prsmd_class import ParseMDClass

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory, RedisChatMessageHistory
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_milvus.vectorstores import Milvus

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import tool

from typing import Optional


######################### VARIABLES DE ENTORNO #########################
QUERY = "Tratamiento del parkinson segun el documento aportado"

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSION_EMBEDDING = 3072

INPUT_FILE = "./colonoscopy.txt"
OUTPUT_FILE = f"{INPUT_FILE}_output.txt"
BD_NAME = "EPID"
COLLECTION_NAME = "EPID"

URI_CONNECTION = "http://localhost:19530"
HOST = "localhost"
PORT = 19530


######################### FUNCIONES #########################


def getVectorStoreMilvus(dbName, collectionName, api_key_openAI):
    ######################### CONEXIÓN A MILVUS #########################

    uri = URI_CONNECTION

    client = MilvusClient(
        uri=uri,
        token="joaquin:chamorro"
    )

    connections.connect(alias="default", host=HOST, port=PORT)

    ######################### CREAR LA BASE DE DATOS EN MILVUS #########################

    db_name = dbName
    if db_name not in db.list_database():
        db.create_database(db_name, using="default")
        db.using_database(db_name, using="default")
    else:
        db.using_database(db_name, using="default")

    print(f"Conectado a la base de datos {db_name}")

    ######################### GUARDAR LOS VECTORES EN MILVUS - VECTORSTORE #########################
    #vector_store = Milvus()

    # Crear la función de embeddings de código abierto
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key_openAI)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64}
    }

    vector_store = Milvus(
        embeddings,
        collection_name=collectionName,
        connection_args={"uri": uri},
        enable_dynamic_field=True,
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        index_params=index_params
    )
    print("Colección ya existe")

    print(f"Conexión a Milvus-VectorStore establecida.\nConectado a colleccion: {collectionName}\n")
    
    return vector_store


def getAnswer(query, vector_store, api_key_openAI):
    
    model = ChatOpenAI(api_key=api_key_openAI, model=LLM_MODEL)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10})
        #search_type="similarity", search_kwargs={"k": 10, "filter": {"chapter": "30"}})
        #search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})


    ######################### EJECUTAR EL PIPELINE #########################

    template =  """
                - Contesta como un profesional medico: {context}
                - Si no se aportan documentos:
                    - Menciona que no se aportan documentos
                    - Responde con tu conocimiento
                - Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()})
    chain = setup_and_retrieval | prompt | model | output_parser
    respuesta=chain.invoke(query)

    return respuesta

def update_redis_state():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    st.session_state.redisName = f"{current_time}:DB_{st.session_state.db_select}:Col_{st.session_state.collection}"
        
    
    
def main():
    ######################### OBTENER API KEY DE OPENAI #########################
    # Carga las variables de entorno desde un archivo .env
    load_dotenv(find_dotenv(), override=True)

    # Obtiene la API key de OpenAI desde las variables de entorno
    api_key_openAI = os.environ.get("OPENAI_API_KEY")
    print(api_key_openAI)

    #vector_store = getVectorStoreMilvus(BD_NAME, COLLECTION_NAME, api_key_openAI)

    
    st.title("RAG PROGRAM")
    st.sidebar.markdown("<H1 style='text-align: left'> Panel principal </H1>", unsafe_allow_html=True)
    radioselected = st.sidebar.radio('Selecciona una opción', ['Embeddings', 'RAG', 'Coversations'])

    # Main panel
    if radioselected == 'Embeddings':
        with st.form(key='embeddings_form', clear_on_submit=True):
            st.write('Embeddings')
            st.session_state.uploaded_file = st.file_uploader("Elige un archivo PDF", type=["pdf"], accept_multiple_files=False)
            if st.session_state.uploaded_file is not None:
                st.write("filename:", st.session_state.uploaded_file.name)
                
            col1a, col2a = st.columns(2)
            st.session_state.db_name = col1a.text_input('Enter database')
            st.session_state.collection_name = col2a.text_input('Enter collection')
            
            submitted = st.form_submit_button('Create Embeddings')
            
            if submitted:
                with st.spinner("Creating embeddings... This process may take more than 1 minute"):
                    parseEmbeddding = ParseMDClass(
                        pdf_path=st.session_state.uploaded_file.name,
                        output_path=f"{st.session_state.uploaded_file.name}_output.txt",
                        uri_connection="http://localhost:19530",
                        host="localhost",
                        port=19530,
                        db_name=st.session_state.db_name,
                        collection_name=st.session_state.collection_name,
                    )

                    parseEmbeddding.run()
                    st.write("Embeddings created")

    if radioselected == 'RAG':
        cl1, cl2 = st.columns(2)
        try:
        # Conectar a Milvus
            connections.connect(
                uri="http://localhost:19530",
                token="joaquin:chamorro",
                alias="default"
            )
            cl1.success("Conexión exitosa a Milvus")
        except Exception as e:
            cl1.error(f"Error connecting to Milvus: {e}")
            return

        try:
        #conectar a Redis
            r = redis.Redis(host='localhost', port=6377, db=0)
            cl2.success("Conexión exitosa a Redis")
        except Exception as r:
            cl2.error(f"Error connecting to Redis: {r}")
            

        def on_change_wrapper():
            cliente = MilvusClient(uri=URI_CONNECTION)
            #cliente = get_client(st.session_state.db_select, URI_CONNECTION)
            cliente.using_database(db_name=st.session_state.db_select, using="default")
            collection = cliente.list_collections()
            st.session_state.collections = collection
        
        def on_change_wrapper2():
            cliente = MilvusClient(uri=URI_CONNECTION)
            #cliente = get_client(st.session_state.db_select, URI_CONNECTION)
            cliente.using_database(db_name=st.session_state.my_selection, using="default")
            collection = cliente.list_collections()
            st.session_state.collections = collection    
            
        ######################### CONECTAR A MILVUS (RECUPERAR NOMBRE DB Y COLECCIONES) #########################
        # Obtener lista de bases de datos    
        list_databases = db.list_database()
        list_databases.sort()
        
        if "db_select" not in st.session_state:
            st.session_state.db_select = list_databases[0]
            on_change_wrapper()
        
        st.session_state.db_select = cl1.selectbox('Select database', list_databases, key="my_selection", on_change=on_change_wrapper2)
        
        st.session_state.collection = cl2.selectbox('Select collection', st.session_state.collections)

        if "redisName" not in st.session_state:
            update_redis_state()
            st.session_state.col = st.session_state.collection
        elif st.session_state.col != st.session_state.collection:
            update_redis_state()
            
        st.session_state.redisName = st.text_input("Enter a name for the chat history", value=st.session_state.redisName)


        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            vector_store = getVectorStoreMilvus(st.session_state.db_select, st.session_state.collection, api_key_openAI)
            response = getAnswer(prompt, vector_store, api_key_openAI)
            # Display assistant response in chat message container
            st.session_state.conversation = []
            
            with st.chat_message("assistant"):
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                print("Chat history updated")
                for i in range(len(st.session_state.messages)):
                    #print("-------ROLE-------")
                    #print(st.session_state.messages[i]["role"])
                    #print("-------CONTENT-------")
                    #r.set(f"{st.session_state.redisName}_{i}_role", st.session_state.messages[i]["role"])
                    #print(st.session_state.messages[i]["content"])
                    #r.set(f"{st.session_state.redisName}_{i}_content", f'{st.session_state.messages[i]["role"]}\n{st.session_state.messages[i]["content"]}')
                    #print("################")
                    
                    #print(st.session_state.conversation)
                    if st.session_state.messages[i]["role"] == "user":
                        prompt_user = "##### ***USUARIO***"
                    else:
                        prompt_user = "##### ***CHATGPT***"
                    #print("################")
                    st.session_state.conv = f'{prompt_user}:  \n{st.session_state.messages[i]['content']}  \n'
                    st.session_state.conversation.append(st.session_state.conv)
                    #r.set(f"{st.session_state.redisName}_{i}, f'{prompt_user}, {st.session_state.messages[i]["content"]}'")
 
                    r.set(f"{st.session_state.redisName}", f"{st.session_state.conversation}")
                    #print("***********************")
                    #print(f'{st.session_state.conversation}\n')
                    #r.set(st.session_state.conversation)
                    
    
    if radioselected == 'Coversations':
        try:
            # Conexión a Redis
            r = redis.Redis(host='localhost', port=6377, db=0)
            
            # Verificar conexión
            r.ping()
            print("Conexión a Redis exitosa")
        except redis.ConnectionError as e:
            print(f"Error de conexión a Redis: {e}")
            
        # Obtener todas las claves que coincidan con un patrón
        keys = r.keys()  # Obtiene todas las claves

        keys.sort()
            # Imprimir todas las claves
        listRedis = []
        for key in keys:
            listRedis.append(key.decode('utf-8'))
        st.session_state.listRedis = listRedis
        st.session_state.redis = st.selectbox('Select collection', st.session_state.listRedis)

        colbt1, colbt2 = st.columns(2)
        
        if colbt1.button("Show conversation"):
            r = redis.Redis(host='localhost', port=6377, db=0)
            conversation = r.get(st.session_state.redis)
            txt= conversation.decode('utf-8')
            #st.write(txt)
            # Unimos los textos para estructurarlos adecuadamente
    

            # Convertir el texto en una lista
            listas = ast.literal_eval(txt)
            for lista in listas:
                st.write(lista)
        
        if colbt2.button("Convert to PDF"):
            r = redis.Redis(host='localhost', port=6377, db=0)
            conversation = r.get(st.session_state.redis)
            txt= conversation.decode('utf-8')
            #st.write(txt)
            # Unimos los textos para estructurarlos adecuadamente
    

            # Convertir el texto en una lista
            listas = ast.literal_eval(txt)
            for lista in listas:
                st.write(lista)
            # Unimos los textos para estructurarlos adecuadamente
            txt = '\n'.join(listas)
            st.write(txt)
            # Convertir a PDF
            current_time = datetime.datetime.now().strftime("%Y-%m-%d")
            fName = st.session_state.redis.split('_')

            # Obtener la última parte
            fileName = fName[-1]
            
            converter = MarkdownToPDFConverter(txt, f"{fileName}_{current_time}.pdf")
            converter.convertir_a_pdf()
    
if __name__ == "__main__":
    main()