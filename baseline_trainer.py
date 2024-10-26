import self.logger
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from baseline_agent import OntologyService, LLMService, FakeNewsAgent  
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import requests
import zipfile
from groq import Groq
import logging
import yaml
from datetime import datetime

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
    def load_liar_dataset(data_dir: str = "datasets/liar") -> Tuple[List[Dict], List[Dict]]:
        """
        Load LIAR dataset (Wang, 2017)
        Contains 12.8K human-labeled short statements from politifact.com
        
        Args:
            data_dir: Directory where the dataset is stored
        Returns:
            Tuple of (training_data, test_data)
        """
        # Ensure directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Download if not present
        train_file = f"{data_dir}/train.tsv"
        test_file = f"{data_dir}/test.tsv"
        
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print("Downloading LIAR dataset...")
            url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
            response = requests.get(url)
            zip_path = f"{data_dir}/liar_dataset.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        
        # Load data
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 
                  'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 
                  'mostly_true_counts', 'pants_on_fire_counts', 'context']
        
        train_df = pd.read_csv(train_file, sep='\t', names=columns)
        test_df = pd.read_csv(test_file, sep='\t', names=columns)
        
        # Convert to binary labels (True/False)
        def binarize_label(label: str) -> bool:
            return label in ['true', 'mostly-true', 'half-true']
        
        # Convert to required format
        def convert_to_format(df: pd.DataFrame) -> List[Dict]:
            return [
                {
                    'text': row['statement'],
                    'is_true': binarize_label(row['label']),
                    'metadata': {
                        'speaker': row['speaker'],
                        'context': row['context'],
                        'subject': row['subject']
                    }
                }
                for _, row in df.iterrows()
            ]
        
        return convert_to_format(train_df), convert_to_format(test_df)

    @staticmethod
    def load_fnid_dataset(data_dir: str = "datasets/fnid") -> Tuple[List[Dict], List[Dict]]:
        """
        Load Fake News Inference Dataset (FNID)
        Contains news articles with stance and inference labels
        
        Args:
            data_dir: Directory where the dataset is stored
        Returns:
            Tuple of (training_data, test_data)
        """
        # Download if not present
        os.makedirs(data_dir, exist_ok=True)
        dataset_file = f"{data_dir}/fnid_dataset.csv"
        
        if not os.path.exists(dataset_file):
            print("Downloading FNID dataset...")
            url = "https://raw.githubusercontent.com/MickeysClubhouse/COVID-19-FNID/master/FNID_dataset.csv"
            response = requests.get(url)
            with open(dataset_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        # Load data
        df = pd.read_csv(dataset_file)
        
        # Convert to required format
        data = [
            {
                'text': row['text'],
                'is_true': row['label'] == 1,
                'metadata': {
                    'source': row['source'],
                    'date': row['date']
                }
            }
            for _, row in df.iterrows()
        ]
        
        # Split into train/test
        return train_test_split(data, test_size=0.2, random_state=42)

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
        df = pd.concat([true_df, fake_df]).sample(n=3, random_state=42)
        
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

    def train(self, epochs: int = 5) -> None:
        """
        Train the agent for specified number of epochs.
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_metrics = {
                'epoch': epoch + 1,
                'accuracy': [],
                'confidence': [],
                'hyperparameters': {}
            }
            
            # Train on each item in training set
            for item in self.training_data:
                result = self._process_single_item(item)
                epoch_metrics['accuracy'].append(
                    1 if result['predicted'] == item['is_true'] else 0
                )
                epoch_metrics['confidence'].append(result['confidence'])
            
            # Record hyperparameters after training
            epoch_metrics['hyperparameters'] = self.agent.hyperparameters.copy()
            
            # Calculate epoch statistics
            epoch_metrics['avg_accuracy'] = sum(epoch_metrics['accuracy']) / len(epoch_metrics['accuracy'])
            epoch_metrics['avg_confidence'] = sum(epoch_metrics['confidence']) / len(epoch_metrics['confidence'])
            
            self.training_results.append(epoch_metrics)
            self.logger.info(f"Epoch {epoch + 1} - Accuracy: {epoch_metrics['avg_accuracy']:.3f}, "
                        f"Confidence: {epoch_metrics['avg_confidence']:.3f}")

    def evaluate(self) -> Dict:
        """
        Evaluate the trained agent on test data.
        """
        self.logger.info("Starting evaluation on test set")
        
        test_metrics = {
            'accuracy': [],
            'confidence': [],
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        for item in self.test_data:
            result = self._process_single_item(item)
            predicted = result['predicted']
            actual = item['is_true']
            
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
        # Calculate final_accuracy, handling division by zero
        test_metrics['final_accuracy'] = (sum(test_metrics['accuracy']) / len(test_metrics['accuracy'])
                                        if len(test_metrics['accuracy']) > 0 else 0)

        # Calculate avg_confidence, handling division by zero
        test_metrics['avg_confidence'] = (sum(test_metrics['confidence']) / len(test_metrics['confidence'])
                                        if len(test_metrics['confidence']) > 0 else 0)

        # Calculate precision, handling division by zero
        precision_denominator = test_metrics['true_positives'] + test_metrics['false_positives']
        test_metrics['precision'] = (test_metrics['true_positives'] / precision_denominator
                                    if precision_denominator > 0 else 0)

        # Calculate recall, handling division by zero
        recall_denominator = test_metrics['true_positives'] + test_metrics['false_negatives']
        test_metrics['recall'] = (test_metrics['true_positives'] / recall_denominator
                                if recall_denominator > 0 else 0)
        
        self.logger.info(f"Test Results - Accuracy: {test_metrics['final_accuracy']:.3f}, "
                    f"Precision: {test_metrics['precision']:.3f}, "
                    f"Recall: {test_metrics['recall']:.3f}")
        
        return test_metrics

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

    def plot_training_progress(self) -> None:
        """
        Plot training metrics over epochs.
        """
        epochs = range(1, len(self.training_results) + 1)
        accuracies = [r['avg_accuracy'] for r in self.training_results]
        confidences = [r['avg_confidence'] for r in self.training_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracies, 'b-', label='Accuracy')
        plt.plot(epochs, confidences, 'r-', label='Confidence')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_results(self, filename: str = None) -> None:
        """
        Save training results and final hyperparameters to file.
        """
        if filename is None:
            filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            'training_results': self.training_results,
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
                fake_news_trainer.train(epochs=5)
                test_metrics = fake_news_trainer.evaluate()
                fake_news_trainer.plot_training_progress()
                fake_news_trainer.save_results()

        except FileNotFoundError:
                print("ISOT dataset not available locally")

    def evaluate_multiple_datasets(self):
        datasets = {
                'LIAR': {
                'loader': FakeNewsTrainer.load_liar_dataset,
                'available': True
                },
                'FNID': {
                'loader': FakeNewsTrainer.load_fnid_dataset,
                'available': True
                },
                'ISOT': {
                'loader': FakeNewsTrainer.load_isot_dataset,
                'available': True
                }
        }

        # Process each dataset
        for dataset_name, dataset_info in datasets.items():
                print(f"\nProcessing {dataset_name} dataset...")
                
                try:
                        # Load dataset
                        print(f"Loading {dataset_name} dataset...")
                        train_data, test_data = dataset_info['loader']()
                        print(f"{dataset_name} dataset: {len(train_data)} training samples, {len(test_data)} test samples")
                        
                        # Print example
                        print(f"\nExample from {dataset_name} dataset:")
                        print(train_data[0])
                        
                        # Initialize trainer and run training
                        print(f"\nTraining model on {dataset_name} dataset...")
                        fake_news_trainer = FakeNewsTrainer(train_data, test_data)
                        fake_news_trainer.train(epochs=5)
                        
                        # Evaluate and save results
                        test_metrics = fake_news_trainer.evaluate()
                        fake_news_trainer.plot_training_progress()
                        fake_news_trainer.save_results(dataset_name)  # Modified to include dataset name
                        
                        print(f"\nCompleted training and evaluation on {dataset_name} dataset")
                        print("Test metrics:", test_metrics)
                
                except FileNotFoundError:
                        print(f"{dataset_name} dataset not available locally")
                        dataset_info['available'] = False
                        continue
        
                except Exception as e:
                        print(f"Error processing {dataset_name} dataset: {str(e)}")
                        continue

        # Print summary of processed datasets
        print("\nDataset Processing Summary:")
        for dataset_name, dataset_info in datasets.items():
                status = "Processed successfully" if dataset_info['available'] else "Not available"
                print(f"{dataset_name}: {status}")

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

if __name__ == "__main__":
        fake_news_trainer = FakeNewsTrainer()
        fake_news_trainer.evaluate_single_dataset()
        #fake_news_trainer.evaluate_multiple_datasets()