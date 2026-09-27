from pathlib import Path
from loader import LOADER_MAP

BASE_DIR = Path(__file__).parent


def parse_folder(folder_name: str):
    folder_path = BASE_DIR / folder_name
    all_documents = []
    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue
        loader_func = LOADER_MAP.get(file_path.suffix.lower())
        if loader_func is None:
            print(f"Skipping unsupported file: {file_path.name}")
            continue
        print(f"Loading: {file_path.name}")
        all_documents.extend(loader_func(str(file_path)))
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    docs = parse_folder("documents")
    print(docs[0] if docs else "No documents found.")