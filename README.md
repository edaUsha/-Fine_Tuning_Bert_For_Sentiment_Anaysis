# Fine Tuning Bert For Sentiment Anaysis

## Brief

This project fine-tunes BERT (bert-base-uncased) for a **3-class sentiment classification** task using the syedkhalid0/Sentiment-Analysis dataset from HuggingFace, which contains ~105K English text samples labeled as *Negative (0)*, *Neutral (1)*, and *Positive (2)*, split across train (84K), validation (10.5K), and test (10.5K) sets. 

The dataset covers diverse text sources including tweets, product reviews, and movie reviews, pre-processed to remove duplicates and null values for consistency. 

Built using **HuggingFace transformers** and datasets libraries, the pipeline handles tokenization via **AutoTokenizer**, dynamic padding via **DataCollatorWithPadding**.


Training setup:

Tokenization with AutoTokenizer and truncation on the "text" field.

Dynamic padding using DataCollatorWithPadding.

Base BERT parameters frozen, pooler layers unfrozen for efficient fine‑tuning.

Training performed with Trainer and TrainingArguments (learning rate 2e-5, batch size 16, epochs 3) on a GPU runtime.

Deployment:

Model saved to results/ and optionally uploaded to Hugging Face Hub as edaUsha/Fine_Tuning_Bert_For_Sentiment_Anaysis.

Metrics
Evaluation uses a custom compute_metrics function with accuracy_score and roc_auc_score from sklearn.metrics.

Probability scores are obtained by applying softmax to the model logits.

Metrics computed:

Accuracy – fraction of correctly predicted labels.

Macro ROC‑AUC (OvR) – multi‑class ROC‑AUC using one‑vs‑rest averaging.

Example validation results:

python
{'eval_loss': 0.5747562646865845, 'eval_accuracy': 0.758, 'eval_auc': 0.915}
So, on the validation set the model achieves:

Validation loss: 0.5748

Accuracy: 0.758

Macro ROC‑AUC: 0.915


