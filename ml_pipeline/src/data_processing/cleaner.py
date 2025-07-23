import pandas as pd
from sentence_transformers import SentenceTransformer, util
import logging

class DataCleaner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.95):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def clean_text(self, text: str) -> str:
        return ' '.join(text.strip().lower().split())

    def remove_duplicates(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        self.logger.info('Removing duplicates using semantic similarity...')
        texts = df[text_column].tolist()
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        keep = []
        seen = set()
        for idx, emb in enumerate(embeddings):
            if idx in seen:
                continue
            keep.append(idx)
            sims = util.pytorch_cos_sim(emb, embeddings)[0]
            for j, score in enumerate(sims):
                if j != idx and score > self.similarity_threshold:
                    seen.add(j)
        cleaned_df = df.iloc[keep].copy()
        self.logger.info(f'Reduced from {len(df)} to {len(cleaned_df)} rows.')
        return cleaned_df

    def process(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        df[text_column] = df[text_column].astype(str).map(self.clean_text)
        df = self.remove_duplicates(df, text_column)
        return df 