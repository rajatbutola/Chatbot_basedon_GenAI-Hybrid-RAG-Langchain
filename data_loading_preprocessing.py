#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:04:53 2025

@author: karida
"""


from langchain_community.document_loaders import UnstructuredHTMLLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import chardet

loader = UnstructuredHTMLLoader("cleaned_content.html")
data = loader.load()


def detect_encoding(file_path):
    # Open the file in binary mode and read a portion of it
    with open(file_path, "rb") as file:
        raw_data = file.read(10000)  # Read the first 10KB
    # Use chardet to detect the encoding
    result = chardet.detect(raw_data)
    return result["encoding"]

def load_csv_files_with_encoding(folder_path):
    csv_files = glob.glob(f"{folder_path}/*.csv")
    documents = []

    for csv_file in csv_files:
        # Detect the file encoding
        encoding = detect_encoding(csv_file)
        #print(f"Detected encoding for {csv_file}: {encoding}")

        try:
            # Load the CSV with the detected encoding
            loader = CSVLoader(file_path=csv_file, encoding=encoding)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    return documents

# Path to the folder containing CSV files
csv_folder_path = "./output_tables"
documents = load_csv_files_with_encoding(csv_folder_path)
#############################################################

all_documents = data + documents
#############################################################
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000, #upper limit is 1000
    chunk_overlap=200, #chunk will be 1000 including chunk_overlap
    #separators = ["\n\n", "\n"], # seprate the chunk on the basis of given spearators
    length_function=len, 
    is_separator_regex=False,
    add_start_index=True
)
texts = text_splitter.split_documents(all_documents)
len(texts)
#############################################################