.PHONY: run ui test lint eval update-readme

run:
	uvicorn app.main:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501

test:
	pytest -q

lint:
	ruff check .

# Run RAGAS evaluation against the golden dataset.
# Requires: DOC_ID env var (the indexed doc_id for "Attention Is All You Need")
#           GOOGLE_API_KEY env var
# Example:  DOC_ID=abc123 make eval
eval:
	@if [ -z "$(DOC_ID)" ]; then \
		echo "ERROR: DOC_ID is not set.  Run: DOC_ID=<your_doc_id> make eval"; \
		exit 1; \
	fi
	python eval/run.py --doc-id $(DOC_ID)

# Refresh README.md scores from the latest file in eval/results/ (no eval run needed).
update-readme:
	python eval/run.py --update-readme
