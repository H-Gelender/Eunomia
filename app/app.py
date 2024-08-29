import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from GraphEunomia import EunomiaGraph
from dotenv import load_dotenv

def text_references():

    eunomia = st.session_state.eunomia

    ids = eunomia.ids
    books = eunomia.books
    documents = eunomia.documents
    text = ""
    for index in range(len(ids)):
        text+=f"ID: {ids[index]}\nCode: {books[index]}\n\nDocument: {documents[index]}\n\n"

    return text

# Titre de l'application
def main():

    st.title("Eunomia")

    if 'eunomia' not in st.session_state:

        # Initialiser l'instance d'EunomiaGraph dans st.session_state pour la rendre persistante
        load_dotenv(dotenv_path='environements.env')
        Gllm = GoogleGenerativeAI(model="gemini-pro", temperature=0.1)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
        eunomia = EunomiaGraph(Gllm, embeddings)
        st.session_state.eunomia = EunomiaGraph(Gllm, embeddings)

    eunomia = st.session_state.eunomia

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    col1, col2 = st.columns(2)

    with col1:

        conversation_container = st.empty()

        with conversation_container.container():
            for chat in st.session_state.conversation:
                st.markdown(chat)

        st.markdown("<hr>", unsafe_allow_html=True)
        user_input = st.text_input("Vous : ", value=st.session_state.user_input)
        st.markdown("<hr>", unsafe_allow_html=True)

        if user_input:

            # Appeler la méthode run de votre IA et obtenir la réponse
            question = {"question": user_input}
            response = eunomia.run(question)

            st.session_state.conversation.append(f"**Vous :** {user_input}")
            st.session_state.conversation.append(f"**Eunomia :** {response}")

            conversation_container.empty()
            with conversation_container.container():
                for chat in st.session_state.conversation:
                    st.markdown(chat)


    with col2:

        references = text_references()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.text_area("Références", references, height=400, disabled = True)
        st.markdown("<hr>", unsafe_allow_html=True)


if __name__ == '__main__':

    main()
