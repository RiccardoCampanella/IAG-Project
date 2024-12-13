import logging
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from expert_reasoner_agent import OntologyService, LLMService, FakeNewsAgent  
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import requests
import zipfile
from groq import Groq
import logging
import yaml
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class FakeNewsTrainer:
    def __init__(self, training_data: List[Dict]=None, test_data: List[Dict]=None):
        """
        Initialize the trainer with training and test datasets.
        
        Args:
            training_data: List of dicts with keys 'text' and 'is_true'
            test_data: List of dicts with keys 'text' and 'is_true'
        """
        self.training_data = training_data
        self.test_data = test_data
        self.training_results = []
        self.agent = FakeNewsAgent(OntologyService(), LLMService())
        self.logger = self.setup_logger()

        # Initialize Groq client
        self.client = Groq(
            api_key=config['keys']['llm_api_key'],
        )
        
        # Set default config if none provided
        default_config = {
            'model_specs': {
                'model_type': 'mixtral-8x7b-32768',  # or whatever model you prefer
                'temperature': 0.2
            }
        }
        self.config = config if config else default_config
        self.model = self.config['model_specs']['model_type']

    @staticmethod
    def load_isot_dataset(data_dir: str = "isot_news") -> Tuple[List[Dict], List[Dict]]:
        """
        Load ISOT Fake News Dataset
        Contains articles from legitimate and fake news websites
        
        Args:
            data_dir: Directory where the dataset is stored
        Returns:
            Tuple of (training_data, test_data)
        """
        os.makedirs(data_dir, exist_ok=True)
        true_file = f"{data_dir}/True.csv"
        fake_file = f"{data_dir}/Fake.csv"
        
        if not (os.path.exists(true_file) and os.path.exists(fake_file)):
            print("Please download ISOT dataset from: https://www.uvic.ca/engineering/ece/isot/datasets/")
            raise FileNotFoundError("ISOT dataset files not found")
        
        # Load data
        true_df = pd.read_csv(true_file)
        fake_df = pd.read_csv(fake_file)
        
        # Add labels
        true_df['is_true'] = True
        fake_df['is_true'] = False
        
        # Combine datasets
        df = pd.concat([true_df, fake_df]).sample(n=5, random_state=42)
        
        # Convert to required format
        data = [
            {
                'text': row['text'],
                'is_true': row['is_true'],
                'metadata': {
                    'title': row['title'],
                    'subject': row['subject'],
                    'date': row['date']
                }
            }
            for _, row in df.iterrows()
        ]
        
        # Split into train/test
        return train_test_split(data, test_size=0.2, random_state=42)

    def train(self) -> None:
        """
        Train the agent on the training data once with progress bar.
        """
        self.logger.info("Starting training")
        
        training_metrics = {
            'accuracy': [],
            'confidence': [],
            'hyperparameters': {},
            'predictions': [],
            'true_labels': []
        }
        
        # Train on each item in training set once with progress bar
        for item in tqdm(self.training_data, desc="Training Progress"):
            result = self._process_single_item(item)
            training_metrics['accuracy'].append(
                1 if result['predicted'] == item['is_true'] else 0
            )
            training_metrics['confidence'].append(result['confidence'])
            training_metrics['predictions'].append(result['predicted'])
            training_metrics['true_labels'].append(item['is_true'])
        
        # Record hyperparameters after training
        training_metrics['hyperparameters'] = self.agent.hyperparameters.copy()
        
        # Calculate statistics
        training_metrics['avg_accuracy'] = sum(training_metrics['accuracy']) / len(training_metrics['accuracy'])
        training_metrics['avg_confidence'] = sum(training_metrics['confidence']) / len(training_metrics['confidence'])
        
        # Calculate confusion metrics
        tn, fp, fn, tp = confusion_matrix(
            training_metrics['true_labels'], 
            training_metrics['predictions']
        ).ravel()
        
        # Add fake news specific metrics
        total = tp + tn + fp + fn
        training_metrics.update({
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Detection rate
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False alarm rate
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,  # Miss rate
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,  # When we say it's fake, how often are we right
            'false_discovery_rate': fp / (fp + tp) if (fp + tp) > 0 else 0,  # When we say it's fake, how often are we wrong
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True negative rate
            'balanced_accuracy': ((tp/(tp+fn) if (tp+fn) > 0 else 0) + (tn/(tn+fp) if (tn+fp) > 0 else 0)) / 2
        })
        
        self.training_results.append(training_metrics)
        self.logger.info(
            f"Training completed:\n"
            f"Accuracy: {training_metrics['avg_accuracy']:.3f}\n"
            f"Balanced Accuracy: {training_metrics['balanced_accuracy']:.3f}\n"
            f"False Alarm Rate: {training_metrics['false_positive_rate']:.3f}\n"
            f"Miss Rate: {training_metrics['false_negative_rate']:.3f}"
        )

    def evaluate(self) -> Dict:
        """
        Evaluate the trained agent on test data with progress bar.
        """
        self.logger.info("Starting evaluation on test set")
        
        test_metrics = {
            'accuracy': [],
            'confidence': [],
            'predictions': [],
            'true_labels': [],
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        for item in tqdm(self.test_data, desc="Evaluation Progress"):
            result = self._process_single_item(item)
            predicted = result['predicted']
            actual = item['is_true']
            
            # Store predictions and labels
            test_metrics['predictions'].append(predicted)
            test_metrics['true_labels'].append(actual)
            
            # Update metrics
            test_metrics['accuracy'].append(1 if predicted == actual else 0)
            test_metrics['confidence'].append(result['confidence'])
            
            # Update confusion matrix
            if actual and predicted:
                test_metrics['true_positives'] += 1
            elif actual and not predicted:
                test_metrics['false_negatives'] += 1
            elif not actual and predicted:
                test_metrics['false_positives'] += 1
            else:
                test_metrics['true_negatives'] += 1
        
        # Calculate final metrics
        tp = test_metrics['true_positives']
        tn = test_metrics['true_negatives']
        fp = test_metrics['false_positives']
        fn = test_metrics['false_negatives']
        total = tp + tn + fp + fn
        
        test_metrics.update({
            'final_accuracy': sum(test_metrics['accuracy']) / len(test_metrics['accuracy']),
            'avg_confidence': sum(test_metrics['confidence']) / len(test_metrics['confidence']),
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'false_discovery_rate': fp / (fp + tp) if (fp + tp) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'balanced_accuracy': ((tp/(tp+fn) if (tp+fn) > 0 else 0) + (tn/(tn+fp) if (tn+fp) > 0 else 0)) / 2
        })
        
        self.logger.info(
            f"Test Results:\n"
            f"Accuracy: {test_metrics['final_accuracy']:.3f}\n"
            f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.3f}\n"
            f"False Alarm Rate: {test_metrics['false_positive_rate']:.3f}\n"
            f"Miss Rate: {test_metrics['false_negative_rate']:.3f}"
        )
        
        return test_metrics

    def plot_training_progress(self) -> None:
        """
        Plot training metrics and confusion matrix.
        """
        if not self.training_results:
            self.logger.warning("No training results to plot")
            return
        
        latest_results = self.training_results[-1]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Performance Metrics
        metrics = ['Accuracy', 'Precision', 'True Positive Rate', 'Specificity']
        values = [
            latest_results['avg_accuracy'],
            latest_results['precision'],
            latest_results['true_positive_rate'],
            latest_results['specificity']
        ]
        
        ax1.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
        ax1.grid(True, axis='y')
        
        # Plot 2: Confusion Matrix
        cm = confusion_matrix(
            latest_results['true_labels'],
            latest_results['predictions']
        )
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['True News', 'Fake News'],
            yticklabels=['True News', 'Fake News'],
            ax=ax2
        )
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot for error analysis
        plt.figure(figsize=(8, 6))
        error_metrics = ['False Positive Rate', 'False Negative Rate', 'False Discovery Rate']
        error_values = [
            latest_results['false_positive_rate'],
            latest_results['false_negative_rate'],
            latest_results['false_discovery_rate']
        ]
        
        plt.bar(error_metrics, error_values, color=['red', 'orange', 'brown'])
        plt.ylim(0, 1.0)
        plt.title('Error Analysis')
        plt.ylabel('Rate')
        plt.tick_params(axis='x', rotation=45)
        for i, v in enumerate(error_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    def summarize_news_into_headline(self, text: str) -> str:
        """
        Summarize a news article text into a concise headline using Groq API.
        
        Args:
            text (str): The full news article text to summarize
        
        Returns:
            str: A concise headline summarizing the main point of the article
        """
        # Truncate text if too long (typical API limits)
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Create the prompt for headline generation
        prompt = (
            "Summarize the following news text into a concise, factual headline "
            "of no more than 10 words. Capture the main point without any speculation.\n\n"
            f"Text: {text}\n\n"
            "Headline:"
        )
        
        try:
            # Create chat completion using Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.2  # Lower temperature for more focused summaries
            )
            
            # Extract and clean the generated headline
            headline = response.choices[0].message.content.strip()
            
            # Remove any common prefixes that might have been generated
            headline = headline.replace("Headline:", "").strip()
            
            self.logger.debug(f"Generated headline: {headline}")
            return headline
            
        except Exception as e:
            self.logger.error(f"Error generating headline: {str(e)}")
            # Return a truncated version of the original text as fallback
            return text[:100] + "..." if len(text) > 100 else text

    def _process_single_item(self, item: Dict) -> Dict:
        """
        Process a single news item through the agent.
        """
        self.agent.current_news_item = self.summarize_news_into_headline(item['text'])
        self.agent.gather_information()
        self.agent.analyze_evidence()
        
        # Extract results
        results = self.agent.analysis_results.get('reasoning_results', {})
        return {
            'predicted': results.get('isTrue', False),
            'confidence': results.get('confidence_percentage', 0) / 100
        }

    def save_results(self, filename: str = None) -> None:
        """
        Save training results and final hyperparameters to file.
        """
        if filename is None:
            filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            'training_metrics': self.training_results[-1] if self.training_results else {},
            'final_hyperparameters': self.agent.hyperparameters,
            'test_metrics': self.evaluate()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Results saved to {filename}")


    def evaluate_single_dataset(self):
        # Load LIAR dataset
        """
        print("Loading LIAR dataset...")
        liar_train, liar_test = FakeNewsTrainer.load_liar_dataset()
        print(f"LIAR dataset: {len(liar_train)} training samples, {len(liar_test)} test samples")

        # Print example
        print("\nExample from LIAR dataset:")
        print(liar_train[0])

        # Load FNID dataset
        print("\nLoading FNID dataset...")
        fnid_train, fnid_test = FakeNewsTrainer.load_fnid_dataset()
        print(f"FNID dataset: {len(fnid_train)} training samples, {len(fnid_test)} test samples")

        # Print example
        print("\nExample from FNID dataset:")
        print(fnid_train[0])
        """

        try:
                # Try loading ISOT dataset if available
                print("\nLoading ISOT dataset...")
                isot_train, isot_test = FakeNewsTrainer.load_isot_dataset()
                print(f"ISOT dataset: {len(isot_train)} training samples, {len(isot_test)} test samples")
                
                print("\nExample from ISOT dataset:")
                print(isot_train[0])

                fake_news_trainer = FakeNewsTrainer(isot_train, isot_test)
                fake_news_trainer.train()

                fake_news_trainer.plot_training_progress()
                fake_news_trainer.save_results()

        except FileNotFoundError:
                print("ISOT dataset not available locally")


    def setup_logger(self):
        """
        Configure logging to write to a file with date-based naming.
        Suppresses terminal output while maintaining detailed logging in files.
        """
        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a date-based log filename
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_filename = os.path.join(log_dir, f'baseline_trainer_{current_date}.log')
        
        # Create a logger instance
        logger = logging.getLogger('FakeNewsTrainer')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Add the file handler to the logger
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
# Setting up seaborn theme for better readability
sns.set_theme()

def plot_test_metrics(metrics_data):
    """
    Plot test metrics, confusion matrix, error analysis, and confidence over time from JSON data.
    """
    # Extract test metrics data
    test_metrics = metrics_data.get("test_metrics")
    if not test_metrics:
        print("No test metrics available to plot.")
        return
    
    # Test performance metrics
    final_accuracy = test_metrics.get("final_accuracy", 0)
    precision = test_metrics.get("precision", 0)
    true_positive_rate = test_metrics.get("true_positive_rate", 0)
    specificity = test_metrics.get("specificity", 0)
    false_positive_rate = test_metrics.get("false_positive_rate", 0)
    false_negative_rate = test_metrics.get("false_negative_rate", 0)
    false_discovery_rate = test_metrics.get("false_discovery_rate", 0)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance Metrics
    metrics = ['Accuracy', 'Precision', 'True Positive Rate', 'Specificity']
    values = [final_accuracy, precision, true_positive_rate, specificity]
    ax1.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Test Performance Metrics')
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
    ax1.grid(True, axis='y')
    
    # Plot 2: Confusion Matrix
    true_labels = test_metrics.get("true_labels", [])
    predictions = test_metrics.get("predictions", [])
    if true_labels and predictions:
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['True', 'False'],
            yticklabels=['True', 'False'],
            ax=ax2
        )
        ax2.set_title('Test Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot for Error Analysis
    plt.figure(figsize=(8, 6))
    error_metrics = ['False Positive Rate', 'False Negative Rate', 'False Discovery Rate']
    error_values = [false_positive_rate, false_negative_rate, false_discovery_rate]
    plt.bar(error_metrics, error_values, color=['red', 'orange', 'brown'])
    plt.ylim(0, 1.0)
    plt.title('Test Error Analysis')
    plt.ylabel('Rate')
    plt.tick_params(axis='x', rotation=45)
    for i, v in enumerate(error_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Confidence Over Time Plot
    confidence_scores = test_metrics.get("confidence", [])
    if confidence_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(confidence_scores)), confidence_scores, color="purple", marker="o")
        plt.title("Test Confidence Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Confidence Score")
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#with open("training_results_20241027_211454.json", "r") as file:
    #metrics_data = json.load(file)
    #plot_test_metrics(metrics_data)

if __name__ == "__main__":
        fake_news_trainer = FakeNewsTrainer()
        fake_news_trainer.evaluate_single_dataset()
