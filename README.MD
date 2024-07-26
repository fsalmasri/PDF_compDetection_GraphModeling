# PDF Symbol Extraction and Deep Learning Model

## Project Overview

This project is divided into two main parts:

1. **Semi-Automatic Extraction of Symbols from PDF Files**
2. **Deep Learning Model for Symbol Detection and Digital Twin Reconstruction**

## Part 1: Semi-Automatic Extraction of Symbols

### Description

The first part of the project involves developing a library that:

- Extracts paths and vectors from PDF files.
- Parses and analyzes these paths and vectors.
- Extracts symbols from the parsed data.

### Label Studio Integration

Once the symbols are extracted, Label Studio is used for cleaning and annotating the data. This involves:

- Creating and managing annotations for the extracted symbols.
- Ensuring the annotations are clean and accurate for training purposes.

## Part 2: Deep Learning Model for Symbol Detection

### Description

The second part of the project focuses on building a YOLO (You Only Look Once) model to:

- Train on the annotated symbols.
- Detect symbols in new PDF files.

### Digital Twin Reconstruction

Using the trained YOLO model, the project aims to:

- Detect symbols in new PDF files.
- Reconstruct the files in a digital twin format.