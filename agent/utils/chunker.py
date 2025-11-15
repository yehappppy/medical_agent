import os
from agent.utils import logger
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker(RecursiveCharacterTextSplitter):

    def __init__(self, **kwargs):
        self._init_tokenizer()
        self.chunk_token_size = int(os.getenv("CHUNK_TOKEN_SIZE", 1024))
        self.chunk_token_overlap = int(os.getenv("CHUNK_TOKEN_OVERLAP", 256))
        super().__init__(chunk_size=self.chunk_token_size,
                         chunk_overlap=self.chunk_token_overlap,
                         **kwargs)

    def _init_tokenizer(self):
        cache_dir = os.getenv("CACHE_DIR", './cache')
        required_files = [
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt"
        ]
        if not all(
                os.path.exists(os.path.join(cache_dir, f))
                for f in required_files):
            logger.info(f"Downloading tokenizer to {cache_dir}")
            snapshot_download(
                repo_id=os.getenv("TOKENIZER", "Qwen/Qwen3-14B"),
                allow_patterns=["*tokenizer*"],
                cache_dir=cache_dir,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)

    def split_text(self, text) -> list[str]:
        sample_text = text[:4000] if len(text) > 4000 else text
        tokens = self.tokenizer.encode(sample_text)
        ratio = len(sample_text) / len(tokens) if len(sample_text) > 0 else 2.8

        char_chunk_size = int(self.chunk_token_size * ratio)
        char_chunk_overlap = int(self.chunk_token_overlap * ratio)

        self._chunk_size = char_chunk_size
        self._chunk_overlap = char_chunk_overlap

        return super().split_text(text)
