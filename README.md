# chroma-curator

**Smart profiler and export tool for ChromaDB vector collections—perfect for edge AI, mobile apps, and privacy-first analytics.**

## Overview

**chroma-curator** is a Python utility for profiling your [ChromaDB](https://www.trychroma.com/) collections and exporting a curated, on-device-friendly dataset for machine learning and semantic search.

- **Profile:** Analyze your ChromaDB for field completeness, duplicates, embedding stats, document richness, and more.
- **Curate:** Export the most relevant, recent, and unique vectors and metadata—ready for iOS, Android, or embedded ML.
- **Simple:** Human-readable output in JSON, ready for Swift, Kotlin, or Python.
- **Extendable:** Easily tweak for other export formats or selection logic.

## Use Cases

- Deploy semantic search, recommendation, or LLM-RAG on mobile/edge, without needing a full ChromaDB backend.
- Audit, visualise, or downsample large vector stores for model evaluation or PII privacy review.
- Ship fast, private, offline-ready vector datasets for ML or app prototyping.

## Features

- Profiles vector collections: counts, embedding dimensions, duplicates, document lengths, top terms, field coverage, and more.
- Exports a JSON file with vectors, metadata, and text, optimised for edge ML.
- Customisable selection: choose by document length, recency, field richness, or any logic.
- Fully commented, production-quality Python code.

## Getting Started


### 1. (Optional) Set Up a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```
### 2. Install Requirements
```
pip install chromadb pandas numpy
```
### 3. Clone the Repo
```
git clone https://github.com/yourusername/chroma-curator.git
cd chroma-curator
```
### 4. Run the Profiler and Export Tool
```
./cli.py
```
By default, it profiles your ChromaDB database at `./chroma/` and exports:

* `chroma_profile.json` – a summary of your dataset
* `export_for_edge.json` – a compact, smart export for mobile/edge use

### 5. Customisation
Change the database path in the script (default: `./chroma`)
Adjust `top_n` for export size (e.g., 4096 for large devices)
Edit `key_fields` to select which metadata you want in your output

## Example Output

```
[
  {
    "id": "CVE-2024-0001",
    "title": "Buffer Overflow in XYZ",
    "summary": "Exploitable buffer overflow affecting ...",
    "vector": [0.123, 0.456, 0.789, ...],
    "document": "Full text of article/threat..."
  },
  ...
]
```

## FAQ

**Q**: Can I export to SQLite/CSV?

**A**: Yes! The code is simple to adapt. Just ask or open an issue.

**Q**: How do I use the output in iOS/Swift/ML?

**A**: Load the JSON, and use the vectors for nearest neighbour search or Core ML. See the repo wiki for Swift examples.

## License

MIT License

## Contributing

PRs and issues welcome! For advanced features (clustering, more formats, etc.) please open an issue or submit a pull request.

Made with ❤️ by the edge AI & data science community.

## **Optional: Shebang for CLI**

You can add a shebang to the top of `cli.py` to allow direct execution:
```
#!/usr/bin/env python3
```
Then make it executable:
```
chmod +x cli.py
```

Now you can run:
```
./cli.py
```