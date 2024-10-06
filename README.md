# KNN Classifier Algorithm for Parking Lot Occupancy Detection

Welcome to the **KNN Classifier** project! This project focuses on using the **Parking Lot (PKLot) Dataset** from the [VRI UFPR](http://web.inf.ufpr.br/vri/databases/), which classifies parking spots as either **occupied** or **empty**.

### 🌟 Project Overview

The aim of this project is to build a machine learning pipeline that analyzes parking lot images, detecting whether a space is vacant or occupied. This is achieved through image processing, feature extraction, and classification using the **K-Nearest Neighbors (KNN)** algorithm.

### 🛠️ Steps Involved

1. **Image Preprocessing**:  
   I began by **cropping** the large PKLot images into smaller images, each representing a single parking space. These cropped images form the input dataset for the classification task.

2. **Feature Extraction with LBP**:  
   For each cropped image, I applied **Local Binary Patterns** with 8 neighbors and a radius of 1 to capture texture information. The resulting **LBP histograms** serve as features that are saved into a CSV file for each university dataset.

3. **Classification**:  
   Finally, I used the **KNN algorithm** to classify each cropped image as either **empty** or **occupied**, based on the extracted features.

### 📂 Dataset
The dataset used comes from the [PKLot dataset](http://web.inf.ufpr.br/vri/databases/), which contains images of parking lots under various conditions. The dataset is organized by universities, and each university's data is processed and stored in separate CSV files.

### 📊 Feature Extraction Details
- **Method**: Local Binary Patterns (LBP)
- **Neighbors**: 8
- **Radius**: 1
- **Output**: Histograms saved to CSV files

### ⚙️ Classifier
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Task**: Classify parking spaces as **empty** or **occupied**
