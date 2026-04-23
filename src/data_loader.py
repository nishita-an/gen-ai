import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_txt(path):

    docs = []

    for root, _, files in os.walk(path):         # ← recursive walk
        for file in files:
            if file.endswith(".txt"):

                with open(
                    os.path.join(root, file),
                    encoding="utf-8"
                ) as f:
                    text = f.read()

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file}
                    )
                )

    return docs


def load_pdf(path):

    docs = []

    for root, _, files in os.walk(path):         # ← recursive walk
        for file in files:
            if file.endswith(".pdf"):

                loader = PyPDFLoader(
                    os.path.join(root, file)
                )

                docs.extend(loader.load())

    return docs


def load_csv(path):

    docs = []

    for root, _, files in os.walk(path):         # ← recursive walk
        for file in files:
            if file.endswith(".csv"):

                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # row level docs (good for filtering queries)
                for _, row in df.iterrows():

                    row_text = "\n".join(
                        [
                            f"{col}: {row[col]}"
                            for col in df.columns
                        ]
                    )

                    docs.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "source": file,
                                "type": "table_row"
                            }
                        )
                    )

                # table level doc (good for aggregation queries)
                table_text = f"""
Table: {file}

Columns:
{", ".join(df.columns)}

Data:
{df.to_string(index=False)}
"""

                docs.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source": file,
                            "type": "table_full"
                        }
                    )
                )

    return docs