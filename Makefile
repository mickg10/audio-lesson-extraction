# Audio Lesson Extraction Makefile

# Configuration
SHELL := /bin/bash
CONDA_ENV := whisperx2
CUDNN_PATH := /home/mickg/miniconda3/envs/whisperx2/lib/python3.10/site-packages/nvidia/cudnn/lib
YT_SAMPLE := https://youtu.be/Hg2j930q2SI
OUTDIR := ./output
PARALLEL := 1

# Auto-detect CUDA
CUDA_AVAILABLE := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)

# Environment setup (detect CUDA and set LD_LIBRARY_PATH if needed)
ifeq ($(CUDA_AVAILABLE), 1)
	DEVICE := cuda
	ENV_SETUP := export LD_LIBRARY_PATH=$(CUDNN_PATH):$$LD_LIBRARY_PATH
else
	DEVICE := cpu
	ENV_SETUP := 
endif

# Default target
all: test sample

# Run unit tests
test:
	@echo "Running all unit tests..."
	$(ENV_SETUP) && conda run -n $(CONDA_ENV) python run_tests.py

# Process a single sample YouTube video
sample: ensure_outdir
	@echo "Processing sample YouTube video..."
	$(ENV_SETUP) && conda run -n $(CONDA_ENV) python decode_whisper.py --outdir $(OUTDIR) \
		--device $(DEVICE) --parallel $(PARALLEL) $(YT_SAMPLE)

# Process a full set of videos (define YOUR_PLAYLIST_URL)
full: ensure_outdir
	@echo "Please define your playlist or multiple videos after the command: make full YT_URLS='url1 url2...'"
	if [ -z "$(YT_URLS)" ]; then \
		echo "Error: No YouTube URLs provided. Use: make full YT_URLS='url1 url2...'"; \
		exit 1; \
	fi
	$(ENV_SETUP) && conda run -n $(CONDA_ENV) python decode_whisper.py --outdir $(OUTDIR) \
		--device $(DEVICE) --parallel $(PARALLEL) $(YT_URLS)

# Ensure output directory exists
ensure_outdir:
	mkdir -p $(OUTDIR)

# Force reprocessing of videos (use with sample or full targets)
force:
	$(eval FORCE_FLAG := --force)

# Clean output files
clean:
	@echo "Cleaning output directory..."
	rm -rf $(OUTDIR)/*

# Install required dependencies
install_deps:
	conda install -n $(CONDA_ENV) --file requirements.txt

# Help target
help:
	@echo "Audio Lesson Extraction Makefile"
	@echo "------------------------------"
	@echo "Available targets:"
	@echo "  all         : Run tests and process a sample video"
	@echo "  test        : Run all unit tests"
	@echo "  sample      : Process a single sample YouTube video"
	@echo "  full        : Process multiple YouTube videos (specify with YT_URLS='url1 url2...')"
	@echo "  force       : Force reprocessing (use with sample or full, e.g., 'make force sample')"
	@echo "  clean       : Remove all output files"
	@echo "  install_deps: Install required dependencies"
	@echo ""
	@echo "CUDA Status: $(if $(filter 1,$(CUDA_AVAILABLE)),Available (using GPU),Not available (using CPU))"
	@echo "Output Directory: $(OUTDIR)"
	@echo "Parallel Processes: $(PARALLEL)"

.PHONY: all test sample full clean ensure_outdir force install_deps help
