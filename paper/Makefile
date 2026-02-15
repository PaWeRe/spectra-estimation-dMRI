# MRM Manuscript Build
# Usage: make [target]

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean quick

# Full build (with references)
all:
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@echo "Build complete: $(MAIN).pdf"

# Quick build (no references)
quick:
	$(LATEX) $(MAIN)
	@echo "Quick build complete: $(MAIN).pdf"

# Word count (approximate)
wordcount:
	@texcount -inc -total $(MAIN).tex 2>/dev/null || echo "Install texcount for word counting"

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk \
	      sections/*.aux
	@echo "Cleaned auxiliary files"

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf
	@echo "Cleaned all output files"
