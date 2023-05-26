# Visual Question Answering

## Installing modules

```python
!pip install datasets
!pip install nltk
!pip install transformers
```

- The **dataset** library is used here to load and process dataset. It contains functions like **load_dataset** and **set_caching_enabled,** that are used in the project.
- The **********nltk********** library stands for ************************************************Natural Language Toolkit.************************************************ In this project, we are using it to get ****************wordnet**************** data which is a large dictionary of words and their meanings and relations.
- The **************************transformers************************** library is used to get models like BERT, ViT. BERT is used to encode questions and ViT is used to encode images.

## Importing Modules

```python
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score, f1_score
```

## Some initial settings

```python
# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

This code sets the HF_HOME environment variable of operating system. Its value is used by datasets to know the location of cached files. The value of the variable is path to the folder.

CUDA_VISIBLE_DEVICES will be another environment variable of operating system. Its value is set to ‘0’ it means that the system will only use the first GPU that the system has.

```python
set_caching_enabled(True)
logging.set_verbosity_error()
```

**set_caching_enabled(True)** function enables caching for the dataset. Caching is a way of saving some files for later use. This line tells the computer to save some files when it loads or processes datasets.

****************logging.set_verbosity_error()**************** function tells the computer to print only those messages or errors that are very serious. Verbosity is the measure of how much detail the messages and errors have.

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
```

This code is checking if there is a device called ******************‘cuda:0’****************** available. This is the device that uses GPU. If there is this device, then use it otherwise it will use another device called **************‘cpu’.************** CPU is slower but more common.

The next thing it will do is that it will print the name of the device that our code is going to use. If it uses a ******************‘cuda:0’****************** device, then it will print the full name of the device.

For example, in our case it is printing

```python
cuda:0
Tesla T4
```

## Computations with the dataset

```python
dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("/content/drive/MyDrive/vqa-daqar/data_train.csv"),
        "test": os.path.join("/content/drive/MyDrive/vqa-daqar/data_eval.csv")
    }
)
```

This code is using a function called **************************load_dataset.************************** This function will take two arguments. The first one is ****“csv”****. This means that the files are in the CSV format. The second argument is a dictionary that maps, ******************“train”****************** and **************“test”************** with their corresponding location. ****************************data_train.csv**************************** contains the training data and ************data_test.csv************ contains the testing data.

```python
with open(os.path.join("/content/drive/MyDrive/vqa-daqar/answer_space.txt")) as f:
    answer_space = f.read().splitlines()
```

This code will open ********************************answer_space.txt******************************** and will store it in a variable called **********answer_space.********** This will store the list of answers that will be later used to compare the original answer with the predicted answer.

```python
dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
            for ans in examples['answer']
        ]
    },
    batched=True
)
```

After this code, the dataset will now contain the same data as before, plus a new column called ****************‘label’****************. The new column will have labels for each example in the dataset. The labels are the numbers that represent the answers for the questions and images. If there are multiple answers available of a given question, we are selecting only the first to avoid ambiguity. 

## showExample() function

```python
from IPython.display import display

def showExample(train=True, id=None):
    if train:
        data = dataset["train"]
    else:
        data = dataset["test"]
    if id == None:
        id = np.random.randint(len(data))
    image = Image.open(os.path.join("/content/drive/MyDrive/vqa-daqar/images", data[id]["image_id"] + ".png"))
    display(image)

    print("Question:\t", data[id]["question"])
    print("Answer:\t\t", data[id]["answer"], "(Label: {0})".format(data[id]["label"]))
```

The showExample() function takes two arguments. The first argument is **********train********** argument that is a boolean type argument. This argument tells the code whether we want to take the data from training dataset or testing dataset. If **************************train is true**************************, then we select the training data else we select the testing data.

The second argument is ********id.******** This argument is optional and if we don’t pass any value in this argument then it will take the random integer between 0 and length of the data and will assign that value to ****id****. 

It will then display the image with this id and print the corresponding question and answer. Along with answer it will print the label of this answer.

## Multimodal Collator Class

```python
@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join("/content/drive/MyDrive/vqa-daqar/images", image_id + ".png")).convert('RGB') for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
            
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }
```

This code defines a class ************************************MultimodalCollator.************************************ It takes two arguments *tokenizer* and *preprocessor*. The *tokenizer* is instance of **************************AutoTokenizer************************** that can encode text inputs into numerical representations. The *preprocessor* is an instance of ****************************************AutoFeatureExtractor**************************************** that can extract features from image inputs. 

This class has three methods: ************************************tokenize_text**, **preprocess_images*** and *******call.******* 

- ****************tokenize_text*** method takes a list of texts and returns a dictionary of *****************************************************input_ids, token_type_ids, and attention_mask tensors***************************************************** that are used as inputs for the text encoder model. ***************attention_mask*************** tell which tokens are important.
- ********************preprocess_images*** method takes a list of image names as inputs and transforms them into numerical representations that can be used by machine learning model. The function opens the images from a specific folder and converts them to RGB format.
- ***************call*************** function first calls tokenize_text and then calls preprocess_images. It returns input_ids, pixel_values and labels.

## Multimodal VQA Model

```python
class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int = len(answer_space),
            intermediate_dim: int = 512,
            pretrained_text_name: str = 'bert-base-uncased',
            pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
```

*********************MultimodalVQAModel********************* is a custom neural network defined by us. 

***nn.Module*** is a base class for all neural network modules in PyTorch. It provides a way to create and organize reusable building blocks for constructing complex neural networks.

*********__init__********* method :

- ***********num_labels*********** is number of possible answers the model can predict. It is the length of answer space.
- ***intermediate_dim*** is the size of the hidden layer that fuses the text and image features.
- ********bert-base-uncased******** is the pretrained text processing model that we are using and **********************************google/vit-base-patch16-224-in21k********************************** is the recent model for image recognition.

```python
self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
```

This code defines a sequence of layers that combine the text and image features into a single vector.

The linear layer takes the concatenation of text and image features and transforms into a vector of *****************intermediate_dim***************** size.

******************************The dropout layer randomly sets some elements of the vector to zero to prevent overfitting.******************************

****************self.classifier**************** defines a linear layer that takes the fused vector and transforms it into a vector of ******num_labels****** size. ********************************************************Each element of the vector represents the probability of answer being correct.********************************************************

**************RELU : RECTIFIED LINEAR UNIT**************

***************self.criterion*************** defines cross entropy loss function that measures how well the model predicts correct answer. It returns a scalar value that represents the scalar value that represents error.

```python
def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)
        
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out
```

The ****************forward**************** function runs when the model is given some inputs. It takes some arguments like — 

- **********input_ids********** is a tensor of integers that represent the text input in numerical form. It is created by tokenizer.
- *************pixel_values************* is tensor of floats that represents the image input in numerical form. It is created by feature extractor.
- **************attention_mask************** is an optional tensor of integers that indicate which parts of text input are relevant.

*********************************************************************************************logits = self.classifier(fused_output)********************************************************************************************* stores vector that predicts probability of each answer being correct.

## Create Multimodal VQA Collator And Model

```python
def createMultimodalVQACollatorAndModel(text='bert-base-uncased', image='google/vit-base-patch16-224-in21k'):
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )

    multi_model = MultimodalVQAModel(pretrained_text_name=text, pretrained_image_name=image).to(device)
    return multi_collator, multi_model
```

The function creates object called tokenizer and preprocessor. It also creates an object called multi_collator which was defined earlier. multi_model is the visual question answering model. This function will return both multi_collator and multi_model as the output of the function.

## WUP Measures

```python
def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return (semantic_field,weight)

    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)

    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0

    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score
```

This function calculates the ******************************************************************************Wu-Palmer Similarity Score***************************************************** between two words or concepts. This is a number between 0 and 1. It indicates how similar the two words or concepts are based on their meanings and relations in WordNet (****large dataset containing words and their meanings)****. 

*****similarity threshold***** is used to adjust the weight of the similarity score based on how high or low it is.

*********************************************************get_semantic_field********************************************************* function takes a word or concept and returns a list of its possible meanings and a weight. Weight is set to 1.0 by default which means we are very confident about the meaning.

************************************global_weight************************************ is a number that indicates how confident we are about the similarity score.

## Batch WUP Measure

```python
def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)
```

This function calculates average Wu-Palmer similarity score for a batch of labels and predictions.

labels is the correct answer and preds is the predicted answer.

## Show Answers

```python
labels = np.random.randint(len(answer_space), size=5)
preds = np.random.randint(len(answer_space), size=5)

def **showAnswers**(ids):
    print([answer_space[id] for id in ids])

showAnswers(labels)
showAnswers(preds)

print("Predictions vs Labels: ", batch_wup_measure(labels, preds))
print("Labels vs Labels: ", batch_wup_measure(labels, labels))
```

This code is doing some testing and demonstration of the ********batch_wup_measure******** function that was defined earlier.

## Compute Metrics

```python
def **compute_metrics**(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }
```

This function is used to evaluate the performance of the model. It takes two arrays as input, one is predicted and one is for correct answers. The predicted answers are **************logits.************** Correct answers are ****************labels.**************** preds will contain the index of the most probable answer.

This function returns Wu-Palmer score, Accuracy and F1 score.

## Training Arguments

```python
args = TrainingArguments(
    output_dir="checkpoint",
    seed=12345, 
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,             # Save only the last 3 checkpoints at any given time while training 
    metric_for_best_model='wups',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    remove_unused_columns=False,
    num_train_epochs=4,
    #fp16=True,
    # warmup_ratio=0.01,
    # learning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    load_best_model_at_end=True,
)
```

***************************TrainingArguments*************************** is a class from transformers library. This object stores some settings and options for training and evaluating the model.

*****seed***** is the nuber that controls the randomness of training process.

## Create and Train Model

```python
def createAndTrainModel(dataset, args, text_model='bert-base-uncased', image_model='google/vit-base-patch16-224-in21k', multimodal_model='bert_vit'):
    collator, model = createMultimodalVQACollatorAndModel(text_model, image_model)
    
    multi_args = deepcopy(args)
    multi_args.output_dir = os.path.join("..", "checkpoint", multimodal_model)
    multi_trainer = Trainer(
        model,
        multi_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    
    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()
    
    return collator, model, train_multi_metrics, eval_multi_metrics
```

This function is the function used for training of our model. 

*************multi_trainer************* is the object of ************Trainer************ class from transformers library. It then calls the train() and evaluate() functions. 

## Some connecting codes
