import streamlit as st
import os
import shutil
import subprocess
import json
import re
import nltk
import nltk.data
import csv
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from collections import Counter
from nltk.corpus import stopwords
import time
import math

# For .docx and image file support
try:
    import docx
    from PIL import Image
    import pytesseract
except ImportError:
    st.warning("Dependencies for .docx and image support not found. Please install them using: pip install python-docx Pillow pytesseract")

# --- NLTK Initialization ---
try:
    # Check if stopwords are already downloaded
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# --- Brain Memory Configuration ---
BRAIN_MEMORY_DIR = "brainmemory"
BRAIN_KNOWLEDGE_JSON = os.path.join(BRAIN_MEMORY_DIR, "knowledge.json")
BRAIN_KNOWLEDGE_CSV = os.path.join(BRAIN_MEMORY_DIR, "knowledge.csv")
SYNTHESIZED_KNOWLEDGE_FILE = os.path.join(BRAIN_MEMORY_DIR, "synthesized_knowledge.json")
DOCUMENTS_DIR = os.path.join(BRAIN_MEMORY_DIR, "documents")
DELETE_PASSWORD = "brainai" # Hardcoded password for deletion

def initialize_brainmemory():
    """Ensures the brainmemory directory and necessary files exist."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    if not os.path.exists(BRAIN_KNOWLEDGE_JSON):
        with open(BRAIN_KNOWLEDGE_JSON, "w") as f:
            json.dump([], f)
    if not os.path.exists(BRAIN_KNOWLEDGE_CSV):
        with open(BRAIN_KNOWLEDGE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['document_title', 'rectified_topics', 'rectified_entities', 'related_documents', 'rectified_summary'])
    if not os.path.exists(SYNTHESIZED_KNOWLEDGE_FILE):
        with open(SYNTHESIZED_KNOWLEDGE_FILE, "w") as f:
            json.dump({}, f)

def delete_all_brainmemory():
    """
    Deletes the entire brainmemory directory and all its contents.
    This provides a clean slate for new document uploads.
    """
    if os.path.exists(BRAIN_MEMORY_DIR):
        shutil.rmtree(BRAIN_MEMORY_DIR)
    initialize_brainmemory()

def get_installed_ollama_models():
    """Retrieves a list of installed Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = []
        for line in result.stdout.splitlines():
            if "NAME" in line and "ID" in line:
                continue
            parts = line.strip().split()
            if parts:
                model_name = parts[0]
                models.append(model_name)
        return models
    except Exception as e:
        st.error(f"Failed to retrieve models: {e}")
        return []

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyPDF2.
    Returns full_text, a PdfReader object, and an error message.
    """
    full_text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        return full_text, reader, ""
    except Exception as e:
        return "", None, f"Failed to extract text from PDF with PyPDF2. Error: {e}"

def extract_text_from_docx(file_path):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return full_text, None, ""
    except Exception as e:
        return "", None, f"Failed to extract text from .docx. Error: {e}"

def extract_text_from_txt(file_path):
    """Extracts text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        return full_text, None, ""
    except Exception as e:
        return "", None, f"Failed to extract text from .txt. Error: {e}"

def extract_text_from_image(file_path):
    """Extracts text from an image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        if not text.strip():
            return "", None, "No text detected in the image using OCR."
        return text, None, ""
    except Exception as e:
        return "", None, f"Failed to extract text from image with Tesseract. Error: {e}"

def extract_text_from_document(file_path, file_type):
    """
    A unified function to extract text from various document types.
    """
    text, reader, error_msg = "", None, ""
    if file_type == "pdf":
        text, reader, error_msg = extract_text_from_pdf(file_path)
    elif file_type == "docx":
        text, reader, error_msg = extract_text_from_docx(file_path)
    elif file_type in ["png", "jpg", "jpeg"]:
        text, reader, error_msg = extract_text_from_image(file_path)
    elif file_type == "txt":
        text, reader, error_msg = extract_text_from_txt(file_path)
    else:
        error_msg = f"Unsupported file type: {file_type}"
    
    return text, reader, error_msg

# --- New Helper Functions for the improved "Fast (Hybrid)" approach ---
def _tokenize_and_filter(text):
    """Tokenizes text, removes stopwords and short words."""
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def _calculate_tfidf(corpus_tokens):
    """Calculates TF-IDF vectors for a list of tokenized documents."""
    doc_freq = Counter()
    for doc in corpus_tokens:
        doc_freq.update(set(doc))

    total_docs = len(corpus_tokens)
    idfs = {word: math.log(total_docs / (doc_freq[word])) for word in doc_freq}
    
    tfidf_vectors = []
    for doc in corpus_tokens:
        tf = Counter(doc)
        tfidf_vec = {word: tf[word] * idfs[word] for word in tf}
        tfidf_vectors.append(tfidf_vec)
    
    return tfidf_vectors, list(doc_freq.keys())

def _calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two TF-IDF vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[word] * vec2[word] for word in intersection])
    
    sum1 = sum([vec1[word]**2 for word in vec1.keys()])
    sum2 = sum([vec2[word]**2 for word in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def generate_fast_patterns(document_title, full_text, existing_knowledge):
    """
    Generates a comprehensive pattern using a fast, hybrid approach with
    TF-IDF, extractive summarization, and cosine similarity.
    """
    # 1. Create the corpus of documents
    corpus_texts = [full_text]
    document_titles = [document_title]
    for pattern in existing_knowledge:
        # Load existing document text to build a full corpus
        doc_path = os.path.join(DOCUMENTS_DIR, pattern['title'])
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                corpus_texts.append(f.read())
                document_titles.append(pattern['title'])
    
    # 2. Tokenize and calculate TF-IDF for the corpus
    corpus_tokens = [_tokenize_and_filter(text) for text in corpus_texts]
    tfidf_vectors, vocab = _calculate_tfidf(corpus_tokens)

    new_doc_tfidf_vec = tfidf_vectors[0]
    existing_docs_tfidf_vectors = tfidf_vectors[1:]
    
    # 3. Topic Extraction (top 5 TF-IDF words)
    sorted_topics = sorted(new_doc_tfidf_vec.items(), key=lambda item: item[1], reverse=True)
    main_topics = [word for word, score in sorted_topics[:5]]
    
    # 4. Entity Extraction (improved regex-based approach)
    # Regex for all-caps words or capitalized phrases.
    entity_regex = r'\b[A-Z]{2,}\b|\b(?:[A-Z][a-z]*\s?)+(?=\s[A-Z]|\.)'
    all_entities = set(re.findall(entity_regex, full_text))
    # Filter out common capitalized words that aren't usually entities
    common_non_entities = {'The', 'A', 'An', 'This', 'It'}
    key_entities = [e for e in all_entities if e not in common_non_entities]
    
    # 5. Extractive Summary Generation (based on topic words)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = sum(1 for word in main_topics if word in sentence.lower())
        if score > 0:
            sentence_scores[i] = score
    
    # Select the top 5 sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
    top_sentence_indices = [i for i, score in sorted_sentences[:5]]
    top_sentence_indices.sort() # Keep original order
    
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    extractive_summary = " ".join(summary_sentences)
    if not extractive_summary:
        extractive_summary = " ".join(sentences[:5]) # Fallback to first 5 sentences
    
    # 6. Related Document Identification (cosine similarity)
    related_documents = []
    if existing_docs_tfidf_vectors:
        for i, existing_vec in enumerate(existing_docs_tfidf_vectors):
            similarity = _calculate_cosine_similarity(new_doc_tfidf_vec, existing_vec)
            # Use a threshold to determine if documents are related
            if similarity > 0.1: # Threshold can be tuned
                related_documents.append(document_titles[i+1]) # +1 because of new doc at index 0
    
    return {
        "title": document_title,
        "rectified_topics": main_topics,
        "rectified_entities": key_entities,
        "related_documents": list(set(related_documents)),
        "rectified_summary": extractive_summary
    }

def get_llm_pattern(document_title, full_text, existing_knowledge, model_name):
    """
    Uses LLM to generate a descriptive, key-value pattern that is easier to parse.
    This is a more robust approach than asking for direct JSON.
    """
    llm = Ollama(model=model_name)
    existing_knowledge_str = json.dumps(existing_knowledge)
    
    prompt = (
        f"You are a meticulous data verifier and a knowledge synthesizer. A new document titled '{document_title}' is being processed. "
        f"Your tasks are: "
        f"1. **Topics**: Identify the 5 most important topics in the document. Provide them as a comma-separated list. "
        f"2. **Entities**: Identify all significant entities (people, places, organizations) mentioned. Provide them as a comma-separated list. "
        f"3. **Connections**: Analyze this new document against the existing knowledge patterns below and identify which of the existing documents are related. Provide a comma-separated list of the filenames of these related documents. "
        f"4. **Summary**: Provide a concise summary of the document (100-200 words). "
        f"The final output must be in a simple key-value format. Do not use JSON or any other special characters like braces. "
        f"The keys must be 'Topics:', 'Entities:', 'Related Documents:', and 'Summary:'.\n\n"
        f"--- Document Content ---\n{full_text[:2000]}...\n\n" # Truncate for efficiency
        f"--- Existing Knowledge Patterns ---\n{existing_knowledge_str}\n\n"
        f"Output in key-value format only:\n"
        f"Topics: \n"
        f"Entities: \n"
        f"Related Documents: \n"
        f"Summary: \n"
    )
    try:
        response = llm(prompt, timeout=120) # Add a 120-second timeout
        return response
    except TimeoutError:
        st.error(f"LLM call timed out after 120 seconds for document '{document_title}'.")
        return None
    except Exception as e:
        st.error(f"Error during LLM pattern generation for document '{document_title}': {e}")
        return None

def parse_llm_pattern(llm_output, document_title):
    """
    Parses the LLM's key-value text output into a structured dictionary.
    """
    if not llm_output:
        return None
    
    pattern = {
        "title": document_title,
        "rectified_topics": [],
        "rectified_entities": [],
        "related_documents": [],
        "rectified_summary": ""
    }
    
    topics_match = re.search(r"Topics: (.*?)(?:\n|Entities:|$)", llm_output, re.DOTALL)
    entities_match = re.search(r"Entities: (.*?)(?:\n|Related Documents:|$)", llm_output, re.DOTALL)
    related_match = re.search(r"Related Documents: (.*?)(?:\n|Summary:|$)", llm_output, re.DOTALL)
    summary_match = re.search(r"Summary: (.*)", llm_output, re.DOTALL)
    
    if topics_match:
        pattern["rectified_topics"] = [t.strip() for t in topics_match.group(1).split(',') if t.strip()]
    if entities_match:
        pattern["rectified_entities"] = [e.strip() for e in entities_match.group(1).split(',') if e.strip()]
    if related_match:
        pattern["related_documents"] = [r.strip() for r in related_match.group(1).split(',') if r.strip()]
    if summary_match:
        pattern["rectified_summary"] = summary_match.group(1).strip()
    
    # Return None if the parsing is completely empty, indicating a bad LLM response
    if not any(pattern.values()):
        return None
        
    return pattern

def synthesize_new_pattern(brain_knowledge, model_name):
    """
    Uses an LLM to synthesize a new, high-level pattern from all stored knowledge.
    """
    if not brain_knowledge:
        return "No knowledge patterns exist to synthesize."

    llm = Ollama(model=model_name)
    all_patterns_str = ""
    for item in brain_knowledge:
        title = item.get("title", "Unknown Title")
        topics = ', '.join(item.get("rectified_topics", []))
        entities = ', '.join(item.get("rectified_entities", []))
        summary = item.get("rectified_summary", "No summary.")
        related = ', '.join(item.get("related_documents", []))
        all_patterns_str += f"Document '{title}': Topics: {topics}. Entities: {entities}. Summary: {summary}. Related: {related}\n\n"

    prompt = (
        f"You are an expert knowledge synthesizer. Your task is to analyze a collection of knowledge patterns "
        f"from various documents. Identify overarching themes, new connections between entities, and "
        f"emerging insights. "
        f"Do not just repeat the individual summaries. Create a new, high-level synthesis of this knowledge. "
        f"Output this synthesized knowledge in a concise, well-structured format. "
        f"Start with a title like 'Synthesized Knowledge Pattern'.\n\n"
        f"--- All Knowledge Patterns ---\n{all_patterns_str}\n\n"
        f"--- Synthesized Knowledge ---\n"
    )
    try:
        synthesized_pattern = llm(prompt, timeout=180) # Add a 180-second timeout
        with open(SYNTHESIZED_KNOWLEDGE_FILE, "w") as f:
            json.dump({"synthesized_pattern": synthesized_pattern}, f, indent=4)
        return synthesized_pattern
    except TimeoutError:
        return "Failed to synthesize new knowledge: LLM call timed out."
    except Exception as e:
        return f"Failed to synthesize new knowledge: {e}"

def handle_question_with_brain(question, current_document_text, brain_knowledge, model_name):
    """
    Handles a user question by first checking for relevant patterns in the brainmemory
    and then synthesizing an answer. This version is smarter about "learning" and application.
    """
    synthesized_pattern_data = None
    try:
        with open(SYNTHESIZED_KNOWLEDGE_FILE, "r") as f:
            content = f.read()
            if content:
                synthesized_pattern_data = json.loads(content).get("synthesized_pattern")
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    brain_summary = "--- Document Knowledge Patterns ---\n"
    for item in brain_knowledge:
        title = item.get("title", "Unknown Title")
        topics = ', '.join(item.get('rectified_topics', []))
        entities = ', '.join(item.get('rectified_entities', []))
        related = ', '.join(item.get('related_documents', []))
        summary = item.get("rectified_summary", "No summary.")
        brain_summary += f"Document '{title}': Topics: {topics}. Entities: {entities}. Related to: {related}. Summary: {summary}\n"
    if synthesized_pattern_data:
        brain_summary += f"\n--- Synthesized Overarching Knowledge ---\n{synthesized_pattern_data}\n"
    
    llm = Ollama(model=model_name)
    
    prompt = (
        f"You are a highly intelligent AI assistant with a 'brain' that contains knowledge from various documents. "
        f"Your task is to answer the following question. Use the provided patterns from your 'brain' and the current document's content to formulate a comprehensive answer. "
        f"Synthesize the information from all relevant sources to provide a complete and accurate response. "
        f"Your 'brain' has learned from these documents and can apply that knowledge. For example, if your 'brain' contains a document about Python programming, you can generate Python code snippets even if they aren't explicitly in the document. "
        f"The final output must be only the answer itself, without any introductory or concluding sentences about the process of synthesis or the sources used. "
        f"Be precise and factual. If you can't find relevant information, use your general knowledge. "
        f"\n\n--- Brain Knowledge ---\n{brain_summary}\n\n"
        f"--- Current Document Content ---\n{current_document_text}\n\n"
        f"--- User Question ---\n{question}\n\n"
        f"Answer:"
    )
    
    try:
        answer = llm(prompt, timeout=120) # Add a 120-second timeout
        return answer
    except TimeoutError:
        return "Sorry, I encountered an error. The LLM call timed out while processing your question."
    except Exception as e:
        st.error(f"Error while processing question with brainmemory: {e}")
        return "Sorry, I encountered an error while trying to answer your question."

def summarize_page(page_data):
    """
    Summarizes a single page's text using the LLM. Designed for parallel processing.
    """
    page_number, page_text, model_name = page_data
    try:
        llm = Ollama(model=model_name)
        prompt = (
            f"Analyze the following text from page {page_number} of a document. "
            f"Generate a detailed and comprehensive summary of 100 to 500 words. "
            f"Ensure all facts and information are correct and consistent with the original context. "
            f"If any information seems contradictory or requires clarification, note it in the summary. "
            f"Do not make up information. Use only the provided context and your general knowledge to rectify."
            f"Text to summarize:\n\n---\n{page_text}\n---\n\nDetailed Summary for Page {page_number}:"
        )
        summary = llm(prompt, timeout=120) # Add a 120-second timeout
        return {"page": page_number, "summary": summary}
    except TimeoutError:
        return {"page": page_number, "summary": f"Error: LLM call timed out for page {page_number}."}
    except Exception as e:
        return {"page": page_number, "summary": f"Error summarizing page {page_number}: {e}"}

def get_selected_summary_parallel(reader, model_name, selected_pages):
    """Orchestrates the parallel summarization of selected pages in a PDF."""
    try:
        page_data_for_processing = []
        skipped_pages = []
        for page_num in selected_pages:
            try:
                page_text = reader.pages[page_num - 1].extract_text()
                if page_text and page_text.strip():
                    page_data_for_processing.append((page_num, page_text, model_name))
                else:
                    skipped_pages.append(page_num)
            except IndexError:
                skipped_pages.append(page_num)
        
        if skipped_pages:
            st.warning(f"Skipped pages with no text: {', '.join(map(str, skipped_pages))}")

        full_summaries = []
        if not page_data_for_processing:
            st.error("No text found in the selected pages to summarize.")
            return []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(summarize_page, page_data): page_data[0] for page_data in page_data_for_processing}
            summary_progress_bar = st.progress(0, text="Summarization progress...")
            for i, future in enumerate(as_completed(futures)):
                summary_result = future.result()
                full_summaries.append(summary_result)
                progress_percent = (i + 1) / len(page_data_for_processing)
                summary_progress_bar.progress(progress_percent, text=f"Summarizing page {summary_result['page']}...")
        
        full_summaries.sort(key=lambda x: x["page"])
        return full_summaries
    except Exception as e:
        st.error(f"Error during parallel summarization: {e}")
        return []

def parse_page_ranges(page_range_str, max_pages):
    """Parses a string of page ranges and returns a sorted list of unique page numbers."""
    selected_pages = set()
    errors = []
    parts = [p.strip() for p in page_range_str.split(',') if p.strip()]
    if not parts:
        return [], "No pages selected."
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end: start, end = end, start
                if start < 1 or end > max_pages:
                    errors.append(f"Invalid range: '{part}'. Pages must be between 1 and {max_pages}.")
                else:
                    selected_pages.update(range(start, end + 1))
            except ValueError:
                errors.append(f"Invalid page range format: '{part}'. Use 'start-end' format.")
        else:
            try:
                page_num = int(part)
                if 1 <= page_num <= max_pages:
                    selected_pages.add(page_num)
                else:
                    errors.append(f"Invalid page number: '{part}'. Pages must be between 1 and {max_pages}.")
            except ValueError:
                errors.append(f"Invalid page number format: '{part}'.")
    if errors: return [], ", ".join(errors)
    return sorted(list(selected_pages)), ""

def create_consolidated_report_llm(chat_history, summaries, model_name):
    """Generates a single, consolidated report using an LLM."""
    raw_data_string = ""
    for chat_entry in chat_history.values():
        raw_data_string += f"--- Document: {chat_entry['title']} Chat History ---\n"
        for chat in chat_entry['conversation']:
            raw_data_string += f"User: {chat['question']}\n"
            raw_data_string += f"Assistant: {chat['answer']}\n\n"
        raw_data_string += "\n"

    if summaries:
        raw_data_string += "--- Page Summaries ---\n"
        for summary_data in summaries:
            raw_data_string += f"Summary for Page {summary_data['page']}:\n"
            raw_data_string += f"{summary_data['summary']}\n\n"

    prompt = (
        f"You are an expert document analyst. Below is a raw log of a conversation about a document, "
        f"along with a series of page-by-page summaries. Your task is to synthesize all of this information "
        f"into a single, comprehensive, and well-organized report. The report should be easy to read and "
        f"should only contain the most relevant insights and information from the raw data. "
        f"Do not include unnecessary conversational fillers or repeated information. "
        f"Start with a high-level overview and then dive into details, integrating key points from "
        f"both the chat and the summaries. Use clear headings to structure your report. "
        f"Focus on consolidating the information to present a clean, concise final document. "
        f"\n\n--- Raw Data ---\n\n{raw_data_string}"
        f"\n\n--- Consolidated Report ---\n\n"
    )
    try:
        llm = Ollama(model=model_name)
        report_content = llm(prompt, timeout=180) # Add a 180-second timeout
        return report_content.encode('utf-8')
    except TimeoutError:
        return "Failed to generate report: LLM call timed out.".encode('utf-8')
    except Exception as e:
        return f"Failed to generate report: {e}".encode('utf-8')

def process_single_document(uploaded_file_data, model_name, processing_method):
    """
    Worker function to process a single document file, extract text,
    generate patterns, and save to brainmemory.
    """
    file_name = uploaded_file_data.name
    file_extension = file_name.split('.')[-1].lower()
    file_path = os.path.join(DOCUMENTS_DIR, file_name)

    try:
        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file_data.getbuffer())

        # Load existing knowledge
        brain_knowledge = []
        try:
            if os.path.exists(BRAIN_KNOWLEDGE_JSON):
                with open(BRAIN_KNOWLEDGE_JSON, "r") as f:
                    content = f.read()
                    if content:
                        brain_knowledge = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # Handle cases where the JSON is malformed or not found
            st.warning(f"Error loading existing knowledge base: {e}. Starting with an empty knowledge base for this session.")
            brain_knowledge = []
        
        # Extract text from the new document
        text, reader, error_msg = extract_text_from_document(file_path, file_extension)
        if not text:
            os.remove(file_path) # Clean up the file
            return file_name, "", None, f"Text extraction failed for '{file_name}': {error_msg}"
        
        # Get pattern based on the selected method
        rectified_pattern = None
        if processing_method == "Intelligent (LLM)":
            llm_output = get_llm_pattern(file_name, text, brain_knowledge, model_name)
            rectified_pattern = parse_llm_pattern(llm_output, file_name)
            if not rectified_pattern:
                status_msg = f"LLM pattern generation failed for '{file_name}'. Falling back to fast pattern extraction."
                rectified_pattern = generate_fast_patterns(file_name, text, brain_knowledge)
            else:
                status_msg = f"A new pattern for document '{file_name}' was successfully created using the LLM."
        else: # "Fast (Hybrid Approach)"
            rectified_pattern = generate_fast_patterns(file_name, text, brain_knowledge)
            status_msg = f"A new pattern for document '{file_name}' was successfully created using the fast hybrid approach."
        
        # Add the new pattern and save to JSON and CSV
        brain_knowledge.append(rectified_pattern)

        try:
            with open(BRAIN_KNOWLEDGE_JSON, "w") as f:
                json.dump(brain_knowledge, f, indent=4)
        except Exception as e:
            return file_name, "", None, f"An error occurred while saving to knowledge.json for '{file_name}': {e}. Pattern NOT saved."
        
        try:
            with open(BRAIN_KNOWLEDGE_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    rectified_pattern.get('title', ''),
                    ', '.join(rectified_pattern.get('rectified_topics', [])),
                    ', '.join(rectified_pattern.get('rectified_entities', [])),
                    ', '.join(rectified_pattern.get('related_documents', [])),
                    rectified_pattern.get('rectified_summary', '')
                ]
                writer.writerow(row)
        except Exception as e:
            return file_name, "", None, f"An error occurred while saving to knowledge.csv for '{file_name}': {e}. Pattern NOT saved."
        
        return file_name, text, reader, status_msg
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return file_name, "", None, f"An unexpected error occurred during processing '{file_name}': {e}"

def clear_chat_history():
    if st.session_state.current_document in st.session_state.chat_history:
        st.session_state.chat_history[st.session_state.current_document]["conversation"] = []
    st.rerun()

# --- Main Streamlit App Logic ---
if __name__ == "__main__":
    st.set_page_config(page_title="Multi-Doc Chatbot with Verified Brainmemory", layout="wide")
    st.title("🧠 THE BRAIN AI")
    st.write("Upload multiple documents to automatically process them in parallel, save their knowledge, and chat with each one.")

    initialize_brainmemory()
    
    # --- Check and fix session state variables ---
    if "model_name" not in st.session_state:
        st.session_state.model_name = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    
    if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, dict):
        st.session_state.chat_history = {}
    
    if "current_document" not in st.session_state:
        st.session_state.current_document = "General"
    if "summaries" not in st.session_state:
        st.session_state.summaries = None
    if "consolidated_report" not in st.session_state:
        st.session_state.consolidated_report = None
    if "synthesized_pattern" not in st.session_state:
        st.session_state.synthesized_pattern = None

    # --- Sidebar for global controls ---
    with st.sidebar:
        st.header("Upload & Model")
        
        models = get_installed_ollama_models()
        if models:
            selected_model = st.selectbox("Select Ollama Model", options=models)
            st.session_state["model_name"] = selected_model
        else:
            st.warning("No models found. Please ensure Ollama is installed and running, and that you have pulled at least one model (e.g., `ollama pull mistral`).")
            st.session_state["model_name"] = None
        
        # Disable processing-related widgets if no model is selected
        processing_method = st.radio(
            "Select Processing Method:",
            ("Intelligent (LLM)", "Fast (Hybrid Approach)"),
            index=0,
            disabled=st.session_state["model_name"] is None,
            help="Intelligent (LLM) processing is more accurate but slower. The Fast Hybrid approach is much quicker and now provides a more comprehensive analysis (TF-IDF, extractive summary, related documents) than the old local method."
        )

        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents", disabled=st.session_state["model_name"] is None):
                st.session_state.uploaded_files = uploaded_files
                st.session_state.processed_documents = {}
                st.session_state.summaries = None
                st.session_state.consolidated_report = None
                
                with st.status(f"Processing {len(uploaded_files)} document(s) in parallel...", expanded=True) as status:
                    if uploaded_files:
                        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                            futures = {executor.submit(process_single_document, file, st.session_state["model_name"], processing_method): file.name for file in uploaded_files}
                            
                            # Use as_completed to update progress as tasks finish
                            for i, future in enumerate(as_completed(futures)):
                                file_name, text, reader, status_msg = future.result()
                                st.info(status_msg)
                                if text:
                                    st.session_state.processed_documents[file_name] = {"text": text, "reader": reader}
                                    if file_name not in st.session_state.chat_history:
                                        st.session_state.chat_history[file_name] = {"title": file_name, "conversation": []}
                                
                                # Update progress bar
                                progress_percent = (i + 1) / len(uploaded_files)
                                status.progress(progress_percent, text=f"Processed {i + 1} of {len(uploaded_files)} documents...")
                        status.update(label="All documents processed!", state="complete")
                    else:
                        st.warning("No files were uploaded to process.")
                        status.update(label="No files to process.", state="complete")

        st.markdown("---")
        st.header("Memory Control")
        
        st.subheader("⚠️ Delete All Data")
        st.write("This action is permanent and requires a password.")
        delete_password = st.text_input("Enter password", type="password", key="delete_password_input")
        
        if st.button("Delete All Documents and Data", key="delete_button", help="This will permanently delete all uploaded documents, chat history, and patterns."):
            if delete_password == DELETE_PASSWORD:
                delete_all_brainmemory()
                st.session_state.uploaded_files = []
                st.session_state.processed_documents = {}
                st.session_state.chat_history = {}
                st.session_state.current_document = "General"
                st.session_state.summaries = None
                st.session_state.consolidated_report = None
                st.session_state.synthesized_pattern = None
                st.success("All documents and data have been deleted.")
                st.rerun()
            else:
                st.error("Incorrect password. Data not deleted.")

        try:
            with open(BRAIN_KNOWLEDGE_JSON, "r") as f:
                brain_knowledge = json.load(f)
                st.success(f"Patterns in Brainmemory: {len(brain_knowledge)}")
        except (FileNotFoundError, json.JSONDecodeError):
            st.info("Patterns in Brainmemory: 0")

    # --- Main content area with tabs ---
    tab1, tab2, tab3 = st.tabs(["Chat", "Knowledge Base", "Summarize & Report"])

    with tab1:
        st.header("Chat with Documents")
        st.write("Select a document to ask questions about or chat with your entire knowledge base using the 'General' option.")

        doc_titles = ["General"] + list(st.session_state.processed_documents.keys())
        st.session_state.current_document = st.selectbox(
            "Select a Document to Chat with", 
            options=doc_titles,
            index=doc_titles.index(st.session_state.current_document) if st.session_state.current_document in doc_titles else 0
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("Clear Chat", on_click=clear_chat_history, help="Clear the conversation history for the current document.")
        
        with col2:
            if st.session_state.current_document in st.session_state.chat_history:
                chat_data_to_export = st.session_state.chat_history[st.session_state.current_document]["conversation"]
                if chat_data_to_export:
                    chat_history_str = ""
                    for chat in chat_data_to_export:
                        chat_history_str += f"User: {chat['question']}\nAssistant: {chat['answer']}\n\n"
                    st.download_button(
                        label="Export Chat History",
                        data=chat_history_str.encode('utf-8'),
                        file_name=f"{st.session_state.current_document}_chat_history.txt",
                        mime="text/plain"
                    )

        current_doc_text = ""
        if st.session_state.current_document != "General":
            current_doc_data = st.session_state.processed_documents.get(st.session_state.current_document, {})
            current_doc_text = current_doc_data.get("text", "")
        
        if "General" not in st.session_state.chat_history:
            st.session_state.chat_history["General"] = {"title": "General", "conversation": []}
        
        current_chat_history = st.session_state.chat_history.get(st.session_state.current_document, {}).get("conversation", [])

        for chat in current_chat_history:
            with st.chat_message("user"):
                st.markdown(chat['question'])
            with st.chat_message("assistant"):
                st.markdown(chat['answer'])

        question = st.chat_input(f"Ask a question about '{st.session_state.current_document}'...", disabled=st.session_state.model_name is None)
        if question and st.session_state.model_name:
            with st.spinner("Thinking and synthesizing information..."):
                brain_knowledge_for_query = []
                try:
                    with open(BRAIN_KNOWLEDGE_JSON, "r") as f:
                        content = f.read()
                        if content:
                            brain_knowledge_for_query = json.loads(content)
                except (json.JSONDecodeError, FileNotFoundError):
                    st.warning("No document patterns found in brainmemory.")

                answer = handle_question_with_brain(question, current_doc_text, brain_knowledge_for_query, st.session_state.model_name)
            
            st.session_state.chat_history[st.session_state.current_document]["conversation"].append({"question": question, "answer": answer})
            st.rerun()

    with tab2:
        st.header("Knowledge Base")
        st.write("Synthesize an overarching pattern from all stored knowledge or view individual patterns for each document. Export the knowledge dataset below.")
        
        col_kb1, col_kb2 = st.columns([1,1])
        with col_kb1:
            st.subheader("Overarching Synthesized Pattern")
            st.write("This pattern combines all document knowledge into a single, high-level summary.")
            if st.button("Synthesize New Knowledge Pattern", disabled=st.session_state.model_name is None):
                try:
                    with open(BRAIN_KNOWLEDGE_JSON, "r") as f:
                        brain_knowledge = json.load(f)
                        if brain_knowledge:
                            with st.spinner("Analyzing all stored patterns and synthesizing a new knowledge pattern..."):
                                st.session_state.synthesized_pattern = synthesize_new_pattern(brain_knowledge, st.session_state.model_name)
                                st.success("New knowledge pattern synthesized and saved!")
                        else:
                            st.warning("No patterns found in brainmemory to synthesize. Please upload and process at least one document.")
                except (FileNotFoundError, json.JSONDecodeError):
                    st.warning("No patterns found in brainmemory to synthesize. Please upload and process at least one document.")
        
        with col_kb2:
            st.subheader("Export Brain Data")
            st.write("Download the structured knowledge dataset as a CSV file.")
            if os.path.exists(BRAIN_KNOWLEDGE_CSV):
                try:
                    with open(BRAIN_KNOWLEDGE_CSV, 'rb') as f:
                        csv_data = f.read()
                    st.download_button(
                        label="Download knowledge.csv",
                        data=csv_data,
                        file_name="knowledge.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Failed to read knowledge.csv: {e}")
            else:
                st.info("No CSV dataset found. Process documents to generate one.")

        if st.session_state.synthesized_pattern:
            st.markdown(st.session_state.synthesized_pattern)
        
        st.markdown("---")
        
        st.subheader("Individual Document Patterns")
        st.write("These are the distinct knowledge patterns and their connections created for each document uploaded.")
        if st.button("Refresh Individual Patterns"):
            if os.path.exists(BRAIN_KNOWLEDGE_JSON):
                try:
                    with open(BRAIN_KNOWLEDGE_JSON, "r") as f:
                        brain_knowledge_list = json.load(f)
                        if brain_knowledge_list:
                            for pattern in brain_knowledge_list:
                                with st.expander(f"**{pattern.get('title', 'Untitled Document')}**"):
                                    st.write(f"**Topics:** {', '.join(pattern.get('rectified_topics', []))}")
                                    st.write(f"**Entities:** {', '.join(pattern.get('rectified_entities', []))}")
                                    st.write(f"**Related Documents:** {', '.join(pattern.get('related_documents', []))}")
                                    st.write(f"**Summary:** {pattern.get('rectified_summary', 'No summary available.')}")
                        else:
                            st.info("No individual patterns found in brain memory.")
                except json.JSONDecodeError:
                    st.error("Error reading brain memory file. It may be corrupted.")
            else:
                st.info("No patterns found in brain memory.")
        
    with tab3:
        st.header("Summarize & Report")
        st.write("Select a PDF document to generate page summaries or a consolidated report. This feature is only available for PDF files.")

        processed_pdf_files = [
            file for file in st.session_state.processed_documents.keys() 
            if file.lower().endswith('.pdf')
        ]
        
        if processed_pdf_files:
            summary_pdf_choice = st.selectbox(
                "Select a PDF for Summary/Report Generation",
                options=processed_pdf_files
            )
            
            summary_option = st.radio(
                "Choose an option:",
                options=["Generate Page Summaries", "Generate Consolidated Report"]
            )
            
            if summary_option == "Generate Page Summaries":
                max_pages = len(st.session_state.processed_documents[summary_pdf_choice]['reader'].pages)
                page_range_str = st.text_input(
                    f"Enter page numbers or ranges (e.g., 1-3, 5, 8-10). Total pages: {max_pages}",
                    "1",
                    key="page_range_input"
                )

                if st.button("Generate Summaries", disabled=st.session_state.model_name is None):
                    if not st.session_state.model_name:
                        st.error("Please select an Ollama model first.")
                    else:
                        selected_pages, error = parse_page_ranges(page_range_str, max_pages)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.summaries = None
                            with st.status("Generating summaries...", expanded=True) as status:
                                summaries_list = get_selected_summary_parallel(
                                    st.session_state.processed_documents[summary_pdf_choice]['reader'],
                                    st.session_state.model_name,
                                    selected_pages
                                )
                                st.session_state.summaries = summaries_list
                                status.update(label="Summaries generated!", state="complete", expanded=False)

            if st.session_state.summaries:
                st.subheader("Generated Summaries")
                for summary_item in st.session_state.summaries:
                    st.markdown(f"### Page {summary_item['page']}\n{summary_item['summary']}")
                
                summaries_text = "\n\n".join(
                    [f"--- Page {s['page']} ---\n{s['summary']}" for s in st.session_state.summaries]
                )
                st.download_button(
                    label="Export Summaries",
                    data=summaries_text.encode('utf-8'),
                    file_name=f"{summary_pdf_choice}_summaries.txt",
                    mime="text/plain"
                )

            if summary_option == "Generate Consolidated Report":
                st.write("This will create a single report from all chat history and available summaries.")
                if st.button("Generate Report", disabled=st.session_state.model_name is None):
                    if not st.session_state.model_name:
                        st.error("Please select an Ollama model first.")
                    else:
                        with st.spinner("Generating consolidated report..."):
                            st.session_state.consolidated_report = create_consolidated_report_llm(
                                st.session_state.chat_history,
                                st.session_state.summaries,
                                st.session_state.model_name
                            )
                        if st.session_state.consolidated_report:
                            st.success("Report generated!")
                            st.download_button(
                                label="Download Consolidated Report",
                                data=st.session_state.consolidated_report,
                                file_name="consolidated_report.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("Failed to generate the consolidated report.")
        else:
            st.info("Please upload and process at least one PDF file to use this feature.")