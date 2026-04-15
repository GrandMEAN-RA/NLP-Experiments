# Hausa Sentiment Analysis

This experiment provides a clean comparative study across three axes: predictive performance, computational cost (training), and deployment 
efficiency (inference), under two distinct hardware regimes. The results are particularly informative for resource-constrained NLP in 
low-resource languages (Hausa).

**Goal:** Compare classical machine learning (Logistic Regression + TF-IDF) against a multilingual transformer (mBERT) and its
light-weight version, DistilBERT, for Hausa sentiment classification under low-resource conditions.

**Research Relevance:** Directly mirrors the sentiment analysis task in my proposed MSc/MPhil/PhD methodology, establishing baseline 
performance and accuracy-efficiency trade-offs.

---

## Dataset

**Source:** `mangaphd/hausaBERTdatatrain` (Hugging Face)

**Size:** ~ 669937,000 Hausa social media comments

**Labels:** Positive / Negative

**Language:** Hausa (native script and Latin alphabet)

**Citation:** @dataset{hausa_sentiments_corpus,
  title={Hausa Sentiment Corpus},
  author={Mich-Seth Owusu},
  year={2025},
  url={https://huggingface.co/datasets/michsethowusu/hausa-sentiments-corpus}
}

---

## Models Compared

| Model | Type | Parameters | Expected Strengths |
|-------|------|------------|---------------------|
| Logistic Regression + TF-IDF | Classical | ~5,000 (features) | Fast inference, low memory, interpretable |
| mBERT (bert-base-multilingual-cased) | Transformer | 178M | Multilingual transfer, higher accuracy |
| dBERT (distilbert-base-multilingual-cased) | Transformer | 178M | Multilingual transfer, higher accuracy |

---

## Methodology

### 1. Data Preprocessing
- Load dataset via `datasets` library. If load fail, load a downloaded copy from local folder.
- Take a random sample of 400 data points for the demo.
- Train/test split (80/20) with stratification
- Text cleaning (remove URLs, special characters, normalise whitespace)

### 2. Classical Model Pipeline
TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))
    ↓
Logistic Regression (C=1.0, max_iter=1000)
    ↓
Predictions + Evaluation

### 3. Transformer Model Pipeline
python
Tokenizer (dBERT/mBERT)
    ↓
Fine-tuning (3 epochs, batch_size=8, learning_rate=2e-5)
    ↓
Evaluation + Inference timing

### 4. Evaluation Metrics
Task performance: Accuracy, F1 (weighted), Precision, Recall

Efficiency: Training time (seconds), Inference latency (ms/sample)

Statistical significance: McNemar's test for accuracy difference

### 5. Experiment Outputs

#### A. Low Resource Environment
- Using device: cpu
- CPU Brand: AMD A4-6210 APU with AMD Radeon R3 Graphics
- Arch: X86_64
- Max Frequency: 1.7970 GHz
- Logical Cores: 4
- Physical Cores: 4

==================================================
CLASSICAL MODEL: Logistic Regression + TF-IDF
==================================================
Performance Metrics:
  Accuracy:  0.5583
  F1 Score:  0.5484
  Precision: 0.5563
  Recall:    0.5583

 Efficiency:
  Training time: 0.04 seconds
  Inference latency: 1.56 ms/sample
  
 Detailed Classification Report:
              precision    recall  f1-score   support

    Negative       0.55      0.40      0.46        57
    Positive       0.56      0.70      0.62        63

    accuracy                           0.56       120
   macro avg       0.56      0.55      0.54       120
weighted avg       0.56      0.56      0.55       120

==================================================
TRANSFORMER MODEL: dBERT
==================================================
Final Performance Metrics:
  Accuracy:  0.5000
  F1 Score:  0.4269
  Precision: 0.5242
  Recall:    0.5000

Efficiency:
  Training time: 2379.65 seconds
  Inference latency: 894.83 ms/sample 
  
==================================================
TRANSFORMER MODEL: mBERT
==================================================
Final Performance Metrics:
  Accuracy:  0.5781
  F1 Score:  0.5716
  Precision: 0.5887
  Recall:    0.5781

Efficiency:
  Training time: 5041.04 seconds
  Inference latency: 1892.36 ms/sample 

#### B. High Resource Environment
- Using device: cpu
- CPU Brand: AMD Ryzen 5 5500U with Radeon Graphics
- Arch: X86_64
- Max Frequency: 2.0960 GHz
- Logical Cores: 12
- Physical Cores: 6

==================================================
CLASSICAL MODEL: Logistic Regression + TF-IDF
==================================================
 Performance Metrics:
  Accuracy:  0.5583
  F1 Score:  0.5484
  Precision: 0.5563
  Recall:    0.5583

 Efficiency:
  Training time: 0.02 seconds
  Inference latency: 0.25 ms/sample

 Detailed Classification Report:
              precision    recall  f1-score   support

    Negative       0.55      0.40      0.46        57
    Positive       0.56      0.70      0.62        63

    accuracy                           0.56       120
   macro avg       0.56      0.55      0.54       120
weighted avg       0.56      0.56      0.55       120

==================================================
TRANSFORMER MODEL: dBERT
==================================================
Final Performance Metrics:
  Accuracy:  0.4688
  F1 Score:  0.3092
  Precision: 0.2307
  Recall:    0.4688

Efficiency:
  Training time: 618.14 seconds
  Inference latency: 129.23 ms/sample 
  
==================================================
TRANSFORMER MODEL: mBERT
==================================================
Final Performance Metrics:
  Accuracy:  0.5781
  F1 Score:  0.5716
  Precision: 0.5887
  Recall:    0.5781

Efficiency:
  Training time: 1722.92 seconds
  Inference latency: 298.96 ms/sample   

#### C. SUMMARY COMPARISON
============================================================
Low Resource
============================================================
                             model  eval_accuracy  eval_f1  eval_precision  eval_recall  training time (s)  inference latency (ms)
                LogisticRegression       0.558333 0.548367        0.556273     0.558333           0.071640                1.286664
distilbert-base-multilingual-cased       0.500000 0.426910        0.524242     0.500000        2379.647193              894.830637
      bert-base-multilingual-cased       0.578125 0.571558        0.588672     0.578125        5041.038840             1892.357860

============================================================
High Resource
============================================================
                             model  eval_accuracy  eval_f1  eval_precision  eval_recall  training time (s)  inference latency (ms)
                LogisticRegression       0.558333 0.548367        0.556273     0.558333           0.016037                0.251105
distilbert-base-multilingual-cased       0.468750 0.309176        0.230655     0.468750         618.135245              129.232265
      bert-base-multilingual-cased       0.578125 0.571558        0.588672     0.578125        1722.923225              298.955664

============================================================
Performance % Improvement with improved Resource
============================================================
                             model  eval_accuracy  eval_f1  eval_precision  eval_recall  training time (s)  inference latency (ms)
                                LR         0.0000   0.0000          0.0000       0.0000            77.6145                 80.4840
                          dBERT/LR       -53.5717 -96.9347       -916.5714     -53.5715            74.0240                 85.5650
                          mBERT/LR         0.0000   0.0000          0.0000       0.0000            65.8220                 84.2040
						
					         dBERT        -6.2500 -27.5570        -56.0030       -6.667            74.0241                 85.6679
                       mBERT/dBERT        40.0000  81.3825        455.6682      40.0000           -58.4880                -82.9860
					 
                             mBERT         0.0000   0.0000          0.0000       0.0000            65.8221                 84.2019
	  
### 6. Visualisation Output
hausa_sentiment_comparison.png – Bar chart comparing accuracy and latency

Saved Model
- models/logisticRegression.pkl
- models/distilbert-base-multilingual-cased.pkl
- models/bert-base-multilingual-cased.pkl

How to Run
1. Install dependencies
bash
pip install pandas numpy scikit-learn transformers datasets torch matplotlib seaborn
2. Run the script
bash
python hausa_sentiment_analysis.py
3. Or use Jupyter notebook
bash
jupyter notebook hausa_sentiment_analysis.ipynb

### 7. INSIGHTS

#### Predictive Performance: Marginal Gains from Deep Models
Across both hardware settings, the ranking of models is consistent:
•	mBERT → best overall performance
•	Logistic Regression (LR) → close second
•	DistilBERT → weakest performance (notably unstable)

Key Observations
- mBERT reflects balanced classification behavior implying that mBERT captures contextual semantics better, confirmed by the confusion matrix:
	- Low false negatives (1)
	- Moderate false positives (3)
- Logistic Regression despite being a linear model performs surprisingly close to mBERT.
- DistilBERT's performance diverges across runs indicating training instability. Confusion matrices show:
	- Severe class imbalance in predictions
	- Tendency toward majority-class bias

#### Computational Cost: Orders of Magnitude Difference
Transformer models are computationally expensive, even on improved CPUs.

Hardware upgrade yields:
	- Significant reduction for deep models
	- No meaningful effect for LR
Scaling behavior:	
	- LR: O(n•d) efficient
	- Transformers: O(n•L²•d) (sequence + attention overhead)

#### Inference Latency: Deployment Bottleneck
LR is ~1000× faster than mBERT. Even with better CPU:
- mBERT still ~300 ms/sample
- DistilBERT ~129 ms/sample

#### Performance–Efficiency Trade-off
- Logistic Regression
	- Slightly lower F1 (~0.55)
	- Near-zero latency
	- Best efficiency-optimal model
- mBERT
	- Highest F1 (~0.57)
	- Very high latency
	- Accuracy-optimal model
- DistilBERT
	- Worse F1 + high latency
	- Dominated solution

#### Hardware Sensitivity Analysis
- Training time reduced significantly (3–4×)
- Inference latency reduced significantly (~6× for mBERT)
- Model accuracy (virtually unchanged)
Hardware affects efficiency, not statistical performance i.e. model quality is data + training regime dependent, not CPU-bound

#### Confusion Matrix-Level Insights
Logistic Regression: High FP and FN implies limited decision boundary flexibility
mBERT: Lower FN impliies better class separation
DistilBERT: Degenerate predictions, especially in high CPU run, indicates optimization failure or underfitting

#### Recommendations
1. Classical models remain competitive for low-resource languages. Simpler models can match transformers when:
	- Data is limited
	- Signals are lexical
2. Transformers require careful tuning
3. For DSS / real-world systems / deployment environments:
	- If latency constraint < 10 ms → Logistic Regression
	- If accuracy is critical → mBERT (with GPU)
	- Avoid DistilBERT unless properly optimized

### 8. Connections to Distributionally Robust Optimization
This experiment aligns strongly with distributional robustness considerations:
- LR’s stability suggests low variance under distribution constraints
- mBERT improves mean performance but introduces:
	- computational risk
	- deployment fragility

In a DRO framing:
- LR corresponds to a robust baseline under ambiguity sets
- mBERT corresponds to a high-capacity model sensitive to distributional shifts
- DistilBERT exhibits instability under resource-induced perturbations

### 8. Final Conclusion
This experiment demonstrates a non-obvious but critical result:
In low-resource NLP settings, efficiency–robustness trade-offs can outweigh marginal accuracy gains from deep models.
- mBERT: best accuracy, worst efficiency
- LR: near-optimal trade-off (practically dominant)
- DistilBERT: underperforms → requires redesign/tuning

Next Steps
- Extend to efficiency analysis

References
HausaBERTa model: Kumshe/Hausa-sentiment-analysis

MasakhaNER dataset: https://github.com/masakhane-io/masakhane-ner

Hugging Face Transformers documentation